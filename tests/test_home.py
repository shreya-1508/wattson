import pandas as pd

from gridcast.home import climate, datacenters, geo, outlook, rates, recs
from gridcast.home.estimate import HomeFacts, estimate, utilities_for_zip
from gridcast.home.usage import UsageModel, monthly_kwh

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = UsageModel().fit(recs.make_synthetic(n=2500))
    return _MODEL


def prices():
    annual = pd.DataFrame({"period": [str(y) for y in range(2014, 2026)], "frequency": "annual",
                           "cents_per_kwh": [11 + 0.35 * i for i in range(12)]})
    monthly = pd.DataFrame({"period": [f"2026-{m:02d}" for m in range(1, 13)], "frequency": "monthly",
                            "cents_per_kwh": 15.5})
    return pd.concat([annual, monthly], ignore_index=True)


def degree_days():
    return pd.DataFrame({"hdd": [850, 700, 520, 260, 90, 10, 0, 0, 40, 230, 480, 790],
                         "cdd": [0, 0, 5, 30, 120, 310, 430, 400, 230, 45, 0, 0]}, index=range(1, 13))


def test_ranges_are_ordered_and_calibrated():
    m = model()
    p = m.predict_one(recs.home_to_row(2, 2000, 7, 5, 5, 3, 4000, 1500))
    assert p["p10"] < p["p50"] < p["p90"]
    assert 70 <= m.evaluation["p10_p90_coverage_pct"] <= 90
    assert m.evaluation["median_abs_pct_error"] < m.evaluation["baseline_median_abs_pct_error"]


def test_monthly_allocation_sums_to_annual_and_peaks_in_winter_for_electric_heat():
    kwh = monthly_kwh(12000, heat_share=0.4, cool_share=0.15, dd=degree_days())
    assert abs(kwh.sum() - 12000) < 1e-6
    assert kwh.idxmax() == 1


def test_bigger_electric_home_costs_more():
    m = model()
    small = m.predict_one(recs.home_to_row(5, 800, 8, 1, 1, 1, 4000, 1500))["p50"]
    big = m.predict_one(recs.home_to_row(2, 3500, 8, 5, 5, 4, 4000, 1500))["p50"]
    assert big > small


def test_rates_and_outlook():
    h = prices()
    assert abs(rates.current_va_price(h) - 15.5) < 1e-9
    now = rates.utility_rate_now(0.14, h)  # 2024 rate scaled by statewide change since 2024
    assert now > 0.14
    g = outlook.scenario_rates(rates.annual_prices(h))
    assert all(0 < v < 0.1 for v in g.values())
    proj = outlook.project(0.15, 10000, {"flat": 0.0, "up": 0.05}, 2026, years=5)
    final = proj[proj["year"] == 2031].set_index("scenario")["annual_cost"]
    assert abs(final["flat"] - 1500) < 1e-6 and final["up"] > final["flat"]


def test_va_utilities_filter():
    df = pd.DataFrame({"zip": [20147, 23219, 10001], "eiaid": [1, 1, 2],
                       "utility_name": ["Dominion", "Dominion", "ConEd"], "state": ["VA", "VA", "NY"],
                       "service_type": "Bundled", "ownership": "IOU",
                       "comm_rate": 0.1, "ind_rate": 0.08, "res_rate": [0.14, 0.14, 0.25]})
    va = rates.load_va_utilities([df])
    assert set(va["zip"]) == {"20147", "23219"}
    assert len(utilities_for_zip(va, "20147")) == 1


def test_parsers():
    place = geo.parse_census({"result": {"addressMatches": [{
        "matchedAddress": "1 MAIN ST, ASHBURN, VA, 20147",
        "coordinates": {"x": -77.49, "y": 39.04},
        "addressComponents": {"zip": "20147", "state": "VA"}}]}})
    assert place.state == "VA" and place.zip == "20147"
    assert geo.parse_census({"result": {"addressMatches": []}}) is None

    dcs = datacenters.parse_overpass({"elements": [
        {"type": "way", "id": 1, "center": {"lat": 39.03, "lon": -77.46}, "tags": {"name": "DC A"}},
        {"type": "node", "id": 2, "lat": 39.0301, "lon": -77.4601, "tags": {"telecom": "data_center"}},
        {"type": "node", "id": 3, "lat": 37.5, "lon": -77.4, "tags": {}},
    ]})
    assert len(dcs) == 2  # first two are the same site
    assert len(datacenters.nearby(dcs, 39.04, -77.49, 5)) == 1


def test_degree_days_from_daily():
    idx = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    temps = pd.Series(55.0, index=idx)
    dd = climate.degree_days_from_daily(temps)
    assert len(dd) == 12 and abs(dd.loc[1, "hdd"] - 310) < 1e-9 and dd["cdd"].sum() == 0


def test_estimate_end_to_end():
    dcs = pd.DataFrame({"name": ["DC A"], "operator": ["X"], "lat": [39.03], "lon": [-77.46]})
    facts = HomeFacts(home_type=2, sqft=2000, year_built=7, heat_fuel=5, water_fuel=5, people=3)
    est = estimate(model(), facts, degree_days(), prices(), "Dominion", 0.14, dcs, 39.04, -77.49, 2026)
    assert est.monthly_bill["p10"] < est.monthly_bill["p50"] < est.monthly_bill["p90"]
    assert abs(est.monthly["kwh"].sum() - est.annual_kwh["p50"]) < 1e-6
    assert len(est.nearby_dcs) == 1
    fallback = estimate(model(), facts, degree_days(), prices(), None, None, dcs, 39.04, -77.49, 2026)
    assert "Virginia average" in fallback.utility


def test_census_point_parse():
    payload = {"result": {"geographies": {
        "States": [{"STUSAB": "VA", "NAME": "Virginia"}],
        "Counties": [{"NAME": "Loudoun County"}],
        "2020 Census ZIP Code Tabulation Areas": [{"ZCTA5": "20147"}],
        "Census Designated Places": [{"BASENAME": "Ashburn"}],
        "County Subdivisions": [{"BASENAME": "Ashburn"}]}}}
    p = geo.parse_census_point(payload, 39.04, -77.49)
    assert p.state == "VA" and p.zip == "20147" and p.lat == 39.04
    assert p.matched_address == "Selected spot in Ashburn, Loudoun County, VA 20147"
    assert geo.parse_census_point({"result": {"geographies": {}}}, 0, 0) is None


def test_nominatim_parse():
    item = {"lat": "37.54", "lon": "-77.43", "address": {
        "house_number": "900", "road": "East Broad Street", "city": "Richmond",
        "state": "Virginia", "ISO3166-2-lvl4": "US-VA", "postcode": "23219"}}
    p = geo.parse_nominatim(item)
    assert p.matched_address == "900 East Broad Street, Richmond, VA 23219" and p.zip == "23219"
    item["address"].update({"state": "Maryland", "ISO3166-2-lvl4": "US-MD"})
    assert geo.parse_nominatim(item) is None


def test_state_vs_national_gap():
    from gridcast.home import impact

    va = pd.Series([10.0, 11.0, 13.0], index=[2019, 2020, 2021])
    us = pd.Series([12.0, 12.6, 13.2], index=[2019, 2020, 2021])  # +10%
    gap = impact.state_vs_national(va, us, 2019, rate_now=0.15, monthly_kwh=1000)
    assert gap.end_year == 2021 and abs(gap.va_change_pct - 30) < 1e-9 and abs(gap.us_change_pct - 10) < 1e-9
    assert abs(gap.bill_now - 150) < 1e-9
    assert abs(gap.bill_if_national - 150 * 1.1 / 1.3) < 1e-9 and gap.gap_dollars > 0


def test_studies_scale_with_usage():
    from gridcast.home import impact

    small, big = impact.personalize(500), impact.personalize(1500)
    assert (big["For your home"] > small["For your home"]).all()
    at_ref = impact.personalize(impact.REFERENCE_KWH).set_index("Source")["For your home"]
    assert any(abs(v - 2.0) < 1e-9 for v in at_ref)        # Dominion's $2 figure at the reference bill
    assert any(abs(v - 10.48) < 1e-9 for v in at_ref)      # $10.48/MWh at 1 MWh/month
    assert all(s.url.startswith("https://") for s in impact.STUDIES)


def test_indexed_series():
    from gridcast.home import impact

    df = impact.indexed({"a": pd.Series([5.0, 10.0], index=[2020, 2021]),
                         "b": pd.Series([2.0, 3.0], index=[2020, 2021])}, 2020)
    assert df.loc[2020].tolist() == [100.0, 100.0] and df.loc[2021, "a"] == 200.0
