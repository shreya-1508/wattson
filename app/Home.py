"""Wattson: search or click any Virginia home to estimate its power bill.

Run from the project folder:  streamlit run app/Home.py
"""
from __future__ import annotations

import html
import json
import sys
from datetime import date
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridcast.config import REPORTS_DIR  # noqa: E402
from gridcast.home import climate, geo, impact  # noqa: E402
from gridcast.home.rates import annual_prices  # noqa: E402
from gridcast.home.estimate import HomeFacts, estimate, utilities_for_zip  # noqa: E402
from gridcast.home.outlook import project  # noqa: E402
from gridcast.home.recs import FUELS, HOME_TYPES, YEAR_BUILT  # noqa: E402
from gridcast.home.setup import FILES  # noqa: E402
from gridcast.home.usage import UsageModel  # noqa: E402

RADIUS_MI = 10
VA_CENTER, VA_ZOOM = (37.9, -78.6), 7
INK, BLUE, AMBER = "#1C2A33", "#2F6F8F", "#E8B100"
RATE_CASE_URL = "https://insideclimatenews.org/news/07012026/virginia-regulators-approve-new-dominion-rates/"

st.set_page_config(page_title="Wattson", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Public+Sans:wght@400;600&display=swap');
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 100%; }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp td, .stApp th { font-family: 'Public Sans', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Archivo', sans-serif; font-stretch: 112%; letter-spacing: -0.01em; }
.wordmark { font-family: 'Archivo', sans-serif; font-stretch: 125%; font-weight: 800; font-size: 2rem;
            color: #1C2A33; line-height: 1; margin: 0; }
.wordmark span { display: inline-block; width: 0.55em; height: 0.55em; border-radius: 50%;
                 background: #E8B100; margin-left: 0.12em; vertical-align: 0.05em; }
.tagline { color: #4A5A63; margin: 0.3rem 0 0.9rem; font-family: 'Public Sans', sans-serif; }
.address { font-family: 'Archivo', sans-serif; font-weight: 700; font-size: 1.15rem; color: #1C2A33; margin: 0.6rem 0 0.1rem; }
.sub { color: #4A5A63; font-size: 0.9rem; margin: 0 0 0.8rem; font-family: 'Public Sans', sans-serif; }
.readout { background: #FFFFFF; border: 1px solid #C9D3CF; border-radius: 6px; padding: 1.1rem 1.25rem 0.9rem; }
.readout-label { font-family: 'Public Sans', sans-serif; font-size: 0.92rem; color: #4A5A63; }
.readout-value { font-family: 'Archivo', sans-serif; font-stretch: 125%; font-weight: 800; font-size: 3.2rem;
                 line-height: 1; color: #1C2A33; margin: 0.25rem 0 0.9rem; font-variant-numeric: tabular-nums; }
.readout-value small { font-size: 1rem; font-weight: 600; font-stretch: 100%; color: #4A5A63; margin-left: 0.3rem; }
.track { position: relative; height: 14px; border-bottom: 2px solid #1C2A33;
         background: repeating-linear-gradient(90deg, #DCE3E0 0 1px, transparent 1px 10%); }
.band { position: absolute; top: 2px; bottom: 2px; background: #2F6F8F; opacity: 0.28; border-radius: 2px; }
.mark { position: absolute; top: -6px; bottom: -2px; width: 3px; background: #1C2A33; transform: translateX(-50%); }
.ends { display: flex; justify-content: space-between; font-family: 'Public Sans', sans-serif;
        font-size: 0.85rem; color: #4A5A63; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }
.facts { margin: 0.9rem 0 0; display: grid; grid-template-columns: auto 1fr; gap: 0.3rem 1rem; font-size: 0.9rem;
         font-family: 'Public Sans', sans-serif; }
.facts dt { color: #4A5A63; } .facts dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
.legend { font-size: 0.85rem; color: #4A5A63; font-family: 'Public Sans', sans-serif; margin-top: 0.3rem; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin: 0 0.3rem 0 0.8rem; vertical-align: -1px; }
</style>
""", unsafe_allow_html=True)


def md(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't render text between two of them as math."""
    return text.replace("$", "\\$")


# ---------- data ----------
@st.cache_resource
def load_model() -> UsageModel:
    return UsageModel.load(FILES["model"])


@st.cache_data
def load_tables():
    utilities = pd.read_csv(FILES["utilities"], dtype={"zip": str})
    prices = pd.read_csv(FILES["prices"], dtype={"period": str})
    dcs = pd.read_csv(FILES["datacenters"]) if FILES["datacenters"].exists() else \
        pd.DataFrame(columns=["name", "operator", "lat", "lon"])
    evaluation = json.loads(FILES["eval"].read_text()) if FILES["eval"].exists() else {}
    us_prices = pd.read_csv(FILES["us_prices"], dtype={"period": str}) if FILES["us_prices"].exists() else None
    return utilities, prices, dcs.fillna({"operator": ""}), evaluation, us_prices


@st.cache_data(ttl=3600, show_spinner=False)
def search_addresses(query: str) -> list:
    """Census exact match for a full address; otherwise OpenStreetMap search for partial ones.
    Errors are raised (not cached) so the page can say the service is down."""
    exact = geo.geocode(query)
    if exact is not None:
        return [exact] if exact.state == "VA" else []
    return geo.suggest(query)


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_lookup(lat: float, lon: float):
    return geo.reverse(lat, lon)


@st.cache_data(show_spinner=False)
def local_degree_days(lat: float, lon: float) -> pd.DataFrame:
    last = date.today().year - 1
    return climate.fetch_degree_days(lat, lon, last - 4, last)


if not all(FILES[k].exists() for k in ("model", "utilities", "prices")):
    st.markdown('<p class="wordmark">Wattson<span></span></p>', unsafe_allow_html=True)
    st.info("Wattson needs a one-time setup. In your project folder, run:\n\n"
            "`python -m gridcast setup-home`\n\nthen refresh this page.")
    st.stop()

model = load_model()
utilities, prices, dcs, evaluation, us_prices = load_tables()
place: geo.Place | None = st.session_state.get("place")


# ---------- map ----------
def build_map(place: geo.Place | None) -> folium.Map:
    center, zoom = ((place.lat, place.lon), 11) if place else (VA_CENTER, VA_ZOOM)
    # OpenStreetMap's standard tiles: free, no API key, attribution required (shown bottom-right).
    # (CARTO's basemaps now require a key.) The tiles are greyed out with CSS so the amber
    # data center markers stand out; markers live in a separate layer, so they keep their color.
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OpenStreetMap", max_zoom=19,
    ).add_to(m)
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-tile-pane { filter: grayscale(1) brightness(1.06) contrast(0.92); }</style>"))
    for row in dcs.itertuples():
        tip = html.escape(row.name) + (f" ({html.escape(row.operator)})" if row.operator else "")
        folium.CircleMarker((row.lat, row.lon), radius=4, color=AMBER, weight=1, fill=True,
                            fill_color=AMBER, fill_opacity=0.85, tooltip=tip).add_to(m)
    if place:
        folium.Circle((place.lat, place.lon), radius=RADIUS_MI * 1609.34, color=BLUE, weight=1.5,
                      fill=False, dash_array="6 6").add_to(m)
        folium.CircleMarker((place.lat, place.lon), radius=9, color="#FFFFFF", weight=3, fill=True,
                            fill_color=INK, fill_opacity=1, tooltip=html.escape(place.matched_address)).add_to(m)
    return m


map_col, panel = st.columns([7, 4], gap="medium")

with map_col:
    key = f"map-{place.lat:.5f}-{place.lon:.5f}" if place else "map-va"
    out = st_folium(build_map(place), key=key, height=760, use_container_width=True,
                    returned_objects=["last_clicked"])
    st.markdown(f'<div class="legend"><span class="dot" style="background:{AMBER}"></span>Data center site '
                f'<span class="dot" style="background:{INK}"></span>Selected home '
                f'<span class="dot" style="background:transparent;border:1.5px dashed {BLUE}"></span>'
                f'{RADIUS_MI}-mile radius</div>', unsafe_allow_html=True)

    click = (out or {}).get("last_clicked")
    if click and click != st.session_state.get("handled_click"):
        st.session_state["handled_click"] = click
        try:
            picked = reverse_lookup(round(click["lat"], 5), round(click["lng"], 5))
        except Exception:
            picked = None
            st.error("The address service didn't respond. Try clicking again.")
        if picked and picked.state == "VA":
            st.session_state["place"] = picked
            st.rerun()
        elif picked:
            st.warning("That spot is outside Virginia. Wattson covers Virginia homes for now.")


# ---------- panel ----------
with panel:
    st.markdown('<p class="wordmark">Wattson<span></span></p>'
                '<p class="tagline">Estimate the power bill for any home in Virginia.</p>',
                unsafe_allow_html=True)

    query = st.text_input("Search a Virginia address", placeholder="Street, city or ZIP, then press Enter",
                          key="address_query").strip()
    if query and query != st.session_state.get("chosen_query"):
        try:
            with st.spinner("Searching..."):
                matches = search_addresses(query)
        except Exception as exc:
            matches = None
            st.error(f"The address service isn't responding ({type(exc).__name__}). "
                     "Check your internet connection, or click a spot on the map instead.")
        if matches == []:
            st.warning("No Virginia matches. Try adding the city or ZIP code.")
        elif matches:
            st.caption("Pick the right address:")
            for i, m in enumerate(matches):
                if st.button(m.matched_address, key=f"match-{i}", use_container_width=True):
                    st.session_state["place"] = m
                    st.session_state["chosen_query"] = query  # hide the list after choosing
                    st.rerun()
    st.caption("Or click anywhere on the map.")

    if place is None:
        st.markdown(f"""
Pick a home to see its estimated monthly bill, how it could change over the next five years,
and how data center growth may be affecting it.

**How Wattson works**
1. You pick a home and adjust its details: size, age, heating.
2. A model trained on about 18,500 surveyed U.S. homes estimates its yearly electricity use,
   adjusted for the weather at that spot.
3. That use is multiplied by the local utility's current average rate to estimate the bill.
4. Below the map, Wattson lays out what's known about how data centers affect Virginia prices.
""")
        st.write(f"The map shows **{len(dcs):,}** data center sites in Virginia, mapped by OpenStreetMap "
                 "volunteers. Not every site is mapped.")
        st.stop()

    st.markdown(f'<p class="address">{html.escape(place.matched_address)}</p>', unsafe_allow_html=True)
    readout_slot = st.container()

    with st.expander("Edit home details"):
        c1, c2 = st.columns(2)
        home_type = c1.selectbox("Type of home", list(HOME_TYPES))
        sqft = c2.number_input("Square feet", min_value=300, max_value=10000, value=1900, step=50)
        year_built = c1.selectbox("Year built", list(YEAR_BUILT), index=6)
        people = c2.number_input("People living there", min_value=1, max_value=12, value=3)
        heat = c1.selectbox("Heating", list(FUELS), help="Most newer Virginia homes heat with an electric heat pump.")
        water = c2.selectbox("Water heater", list(FUELS))
        zip_utils = utilities_for_zip(utilities, place.zip) if place.zip else utilities.iloc[0:0]
        util_name = util_rate = None
        if len(zip_utils):
            names = zip_utils["utility_name"].tolist()
            util_name = st.selectbox("Electric utility", names, disabled=len(names) == 1,
                                     help=f"Utilities serving ZIP {place.zip}. Check your bill if unsure.")
            util_rate = float(zip_utils.loc[zip_utils["utility_name"] == util_name, "res_rate"].iloc[0])

    facts = HomeFacts(HOME_TYPES[home_type], float(sqft), YEAR_BUILT[year_built],
                      FUELS[heat], FUELS[water], int(people))
    try:
        with st.spinner("Pulling five years of local weather..."):
            dd = local_degree_days(round(place.lat, 2), round(place.lon, 2))
    except Exception:
        st.error("The weather service didn't respond, so the estimate can't be calculated. Try again in a minute.")
        st.stop()

    est = estimate(model, facts, dd, prices, util_name, util_rate, dcs, place.lat, place.lon,
                   start_year=date.today().year, radius_mi=RADIUS_MI)

    with readout_slot:
        b = est.monthly_bill
        scale = b["p90"] * 1.15
        pct = lambda v: f"{v / scale * 100:.1f}%"  # noqa: E731
        st.markdown(f"""
<div class="readout">
  <div class="readout-label">Estimated monthly bill, averaged over a year</div>
  <div class="readout-value">&#36;{b['p50']:,.0f}<small>/ month</small></div>
  <div class="track" role="img" aria-label="Likely range &#36;{b['p10']:,.0f} to &#36;{b['p90']:,.0f}, typical &#36;{b['p50']:,.0f}">
    <span class="band" style="left:{pct(b['p10'])}; width:calc({pct(b['p90'])} - {pct(b['p10'])})"></span>
    <span class="mark" style="left:{pct(b['p50'])}"></span>
  </div>
  <div class="ends"><span>Low &#36;{b['p10']:,.0f}</span><span>High &#36;{b['p90']:,.0f}</span></div>
  <dl class="facts">
    <dt>Utility</dt><dd>{html.escape(est.utility)}</dd>
    <dt>Average rate now</dt><dd>{est.rate_now * 100:.1f}¢ per kWh</dd>
    <dt>Yearly use</dt><dd>{est.annual_kwh['p50']:,.0f} kWh</dd>
    <dt>Data centers within {RADIUS_MI} mi</dt><dd>{len(est.nearby_dcs)}</dd>
  </dl>
</div>""", unsafe_allow_html=True)
        st.caption("8 in 10 homes like this one land between low and high. Habits, insulation and "
                   "thermostat settings move a home within that range.")

    t_month, t_future, t_dc, t_how = st.tabs(["Months", "Next 5 years", "Nearby", "Method"])

    with t_month:
        mo = est.monthly.set_index("month")
        st.bar_chart(mo["cost"], color=BLUE, y_label="Estimated bill ($)", x_label="", height=260)
        peak, low = mo["cost"].idxmax(), mo["cost"].idxmin()
        st.write(f"Highest in **{peak}** (about \\${mo.loc[peak, 'cost']:,.0f}), lowest in **{low}** "
                 f"(about \\${mo.loc[low, 'cost']:,.0f}). Heating is about {est.heat_share:.0%} of yearly use "
                 f"and cooling about {est.cool_share:.0%}.")
        with st.expander("How this works"):
            st.write("The model splits yearly use into three parts: an always-on base (fridge, lights, "
                     "electronics), heating that follows cold months, and cooling that follows hot months. "
                     "Heating and cooling are spread across the year using five years of daily temperatures "
                     "at this location. Each month's use times your rate gives that month's estimated bill.")

    with t_future:
        custom = st.slider("Your own assumption: price change per year", -2.0, 10.0, 3.0, 0.5, format="%.1f%%")
        growth = {**est.growth, "Your assumption": custom / 100}
        proj = project(est.rate_now, est.annual_kwh["p50"], growth, date.today().year)
        wide = proj.pivot(index="year", columns="scenario", values="monthly_cost")
        st.line_chart(wide, color=[BLUE, AMBER, INK], y_label="Average monthly bill ($)", height=260)
        final = wide.iloc[-1]
        st.dataframe(pd.DataFrame({
            "Per year": {k: f"{v * 100:+.1f}%" for k, v in growth.items()},
            f"Bill in {wide.index[-1]}": {k: f"${final[k]:,.0f}" for k in growth},
        }), use_container_width=True)
        with st.expander("How this works"):
            st.write("Each line grows today's rate by a steady yearly percentage, holding your usage constant. "
                     "**Long-run trend** is the average yearly change in Virginia's residential price over the "
                     "last 10 years (EIA). **Recent trend** uses the last 3 years, which captures recent rate "
                     "increases. **Your assumption** is the slider. These are scenarios, not predictions: real "
                     "prices change in steps when regulators approve rate cases.")

    with t_dc:
        near = est.nearby_dcs
        if near.empty:
            st.write(f"No mapped data center sites within {RADIUS_MI} miles.")
        else:
            st.write(f"**{len(near)}** mapped sites within {RADIUS_MI} miles; the closest is about "
                     f"**{near['miles'].iloc[0]:.1f} miles** away.")
            st.dataframe(near[["name", "operator", "miles"]].head(15).rename(
                columns={"name": "Site", "operator": "Operator", "miles": "Miles"}),
                hide_index=True, use_container_width=True,
                column_config={"Miles": st.column_config.NumberColumn(format="%.1f")})
        with st.expander("How this works"):
            st.write(f"Sites come from OpenStreetMap, mapped by volunteers, so some are missing. Distances are "
                     f"straight-line. Being close to a data center doesn't change your rate by itself: rates are "
                     f"set across a utility's whole service territory, so a home in Ashburn and one in Richmond "
                     f"on the same Dominion plan pay the same price per kWh.")

    with t_how:
        e = evaluation
        acc = (f" Tested on {e['homes_tested']:,} homes it never saw: typical error **{e['median_abs_pct_error']:.0f}%** "
               f"(vs {e['baseline_median_abs_pct_error']:.0f}% for a per-square-foot estimate); actual use fell "
               f"inside the low-high range for **{e['p10_p90_coverage_pct']:.0f}%** of homes.") if e else ""
        st.markdown(f"""
- **Usage:** machine learning model trained on the EIA's survey of about 18,500 homes with measured
  electricity use, using your home's details and five years of weather at this location.{acc}
- **Range:** the low and high ends are set so that 8 in 10 homes the model has never seen fall inside them.
- **Price:** your utility's 2024 average residential rate (NREL), updated with Virginia's statewide
  price trend (EIA). It includes fixed monthly charges.
- **Map:** addresses from the U.S. Census geocoder and OpenStreetMap; data center sites from OpenStreetMap.

An educational estimate, not a quote. Your actual bill is on your utility account.
""")


# ---------- data centers and your bill (full width) ----------
monthly_use = est.annual_kwh["p50"] / 12
st.markdown("---")
st.header("How data centers may be affecting this bill")
st.write("Virginia has more data centers than anywhere in the world, and they use a lot of power. Whether "
         "that raises your bill depends on which costs they drive and who is required to pay for them. "
         "Below: how a bill is built, what the price data shows, what published studies estimate for a home "
         "like this one, and what regulators are changing. No single number here proves cause and effect.")

st.subheader("How your bill is built")
st.dataframe(pd.DataFrame([
    ("Energy", "Fuel and power plant output, hour by hour",
     "More demand means more expensive plants run more often, which raises wholesale prices."),
    ("Capacity", "Paying plants to be available on the hottest and coldest days",
     "The grid operator buys more capacity when it forecasts more peak demand. Data centers are the biggest "
     "source of forecast growth in the PJM region."),
    ("Transmission", "High-voltage lines that move power across the region",
     "New lines are built to reach data center clusters. Who pays depends on cost-sharing rules, which "
     "Virginia is changing in 2026-2027."),
    ("New power plants", "Generation and storage the utility builds",
     "Utilities plan new plants partly to serve data centers; those costs reach customers through rates."),
    ("Local wires and fixed charges", "Neighborhood poles, wires and meters",
     "Mostly unaffected: large customers usually pay for their own connections."),
], columns=["Part of your bill", "What it pays for", "How data centers can affect it"]),
    hide_index=True, use_container_width=True)
st.caption("It can cut the other way too: big customers also pay toward shared fixed costs, which can hold rates "
           "down for everyone else. Whether data centers pay their full share is what Virginia regulators are "
           "working on now.")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Virginia vs. the rest of the country")
    va_annual = annual_prices(prices)
    if us_prices is None:
        st.write("Run `python -m gridcast setup-home` to download U.S. prices for this comparison.")
    else:
        us_annual = annual_prices(us_prices)
        years = sorted(set(va_annual.index) & set(us_annual.index))
        base = st.select_slider("Compare starting from", options=years[:-2],
                                value=2019 if 2019 in years[:-2] else years[0])
        gap = impact.state_vs_national(va_annual, us_annual, base, est.rate_now, monthly_use)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Virginia price, {base}-{gap.end_year}", f"{gap.va_change_pct:+.0f}%")
        c2.metric("U.S. average price", f"{gap.us_change_pct:+.0f}%")
        sign = "+" if gap.gap_dollars >= 0 else "-"
        c3.metric("Gap for this home", f"{sign}${abs(gap.gap_dollars):,.0f}/mo")
        if gap.gap_dollars > 0:
            st.write(f"If Virginia prices had followed the U.S. average since {base}, this home's bill would be "
                     f"about **\\${gap.bill_if_national:,.0f}** a month instead of **\\${gap.bill_now:,.0f}**.")
        else:
            st.write(f"Virginia prices have risen *less* than the U.S. average since {base}. Following the "
                     f"national trend, this home's bill would be about **\\${gap.bill_if_national:,.0f}** a month "
                     f"instead of **\\${gap.bill_now:,.0f}**.")
        series = {"Virginia price": va_annual, "U.S. average price": us_annual}
        trend_file = REPORTS_DIR / "dom_baseload.csv"
        if trend_file.exists():
            series["Overnight grid demand, Dominion zone"] = pd.read_csv(trend_file, index_col="year")["median_night_min_mw"]
        chart = impact.indexed(series, base)
        st.line_chart(chart, color=[INK, BLUE, AMBER][: chart.shape[1]], y_label=f"{base} = 100", height=280)
        with st.expander("How to read this"):
            st.write("All lines start at 100 in the chosen year, so you can compare growth rates. Comparing "
                     "Virginia with the national average filters out things that hit every state, like "
                     "inflation and nationwide gas price swings. What's left is everything specific to "
                     "Virginia: data center growth, but also its fuel mix, clean-energy laws, storms and the "
                     "timing of rate cases. The amber line is overnight demand in Dominion's zone, the "
                     "project's signal of 24/7 data center load. Lines moving together is a correlation, "
                     "not proof that one causes the other.")

with right:
    st.subheader("What studies estimate, scaled to this home")
    st.caption(f"Figures quoted for a typical bill are scaled to this home's estimated use of "
               f"{monthly_use:,.0f} kWh a month (Virginia's reference bill uses {impact.REFERENCE_KWH:,} kWh). "
               "They measure different things, so don't add them together.")
    for _, r in impact.personalize(monthly_use).iterrows():
        st.markdown(f"**{r['Source']}**  \n"
                    f"About **{md('$')}{r['For your home']:,.2f} a month** for this home. {md(r['What they found'])}  \n"
                    f"<span style='color:#4A5A63;font-size:0.9rem'>Keep in mind: {md(html.escape(r['Keep in mind']))} "
                    f"[Source]({r['url']})</span>", unsafe_allow_html=True)
        st.write("")

    st.subheader("What's changing")
    for when, what, url in impact.POLICY:
        st.markdown(f"**{when}:** {md(what)} [Source]({url})")

st.info("**Before drawing conclusions:** prices rise for many reasons at once, and the studies above measure "
        "different slices of the bill under different assumptions. The honest summary is that data centers "
        "are adding costs to the regional grid, and Virginia is changing its rules so more of those costs land "
        "on data centers rather than households. How that balances out for your bill is still being decided.")
