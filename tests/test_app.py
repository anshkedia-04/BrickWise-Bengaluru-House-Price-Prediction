import unittest

from src.market_data import (
    get_city_benchmarks,
    get_locality_benchmark,
    get_top_localities_summary,
    load_clean_market_data,
    parse_bhk_value,
    parse_sqft_value,
)
from src.predictor import (
    build_sqft_sensitivity,
    format_location_name,
    get_locality_coefficients,
    load_artifacts,
    predict_price,
)


class BrickWiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.data_columns, cls.locations = load_artifacts()
        cls.market_df = load_clean_market_data()

    def test_artifacts_loaded(self) -> None:
        self.assertEqual(len(self.data_columns), 244)
        self.assertEqual(len(self.locations), 241)

    def test_format_location_name_preserves_acronyms(self) -> None:
        self.assertEqual(format_location_name("hsr layout"), "HSR Layout")
        self.assertEqual(format_location_name("1st block jayanagar"), "1st Block Jayanagar")
        self.assertEqual(format_location_name("sarjapur  road"), "Sarjapur Road")
        self.assertEqual(format_location_name("banashankari stage iii"), "Banashankari Stage III")

    def test_predict_price_standard_home(self) -> None:
        res = predict_price(
            model=self.model,
            data_columns=self.data_columns,
            location="whitefield",
            sqft=1350,
            bhk=2,
            bath=2,
        )
        self.assertGreater(res.estimated_lakhs, 40.0)
        self.assertFalse(res.was_clamped)
        self.assertLess(res.range_low_lakhs, res.estimated_lakhs)
        self.assertGreater(res.range_high_lakhs, res.estimated_lakhs)
        self.assertGreater(res.total_acquisition_lakhs, res.estimated_lakhs)
        self.assertEqual(res.advisories, [])

    def test_predict_price_advisories_and_floor_clamping(self) -> None:
        res = predict_price(
            model=self.model,
            data_columns=self.data_columns,
            location="chikka tirupathi",
            sqft=300,
            bhk=5,
            bath=1,
        )
        self.assertGreaterEqual(res.estimated_lakhs, 5.0)
        self.assertTrue(len(res.advisories) >= 1)

    def test_sensitivity_curve_has_selected_point(self) -> None:
        points = build_sqft_sensitivity(
            model=self.model,
            data_columns=self.data_columns,
            location="indira nagar",
            sqft=1600,
            bhk=3,
            bath=3,
        )
        self.assertEqual(len(points), 7)
        self.assertEqual(sum(1 for p in points if p["is_selected"]), 1)

    def test_locality_coefficients_extracted(self) -> None:
        coefs = get_locality_coefficients(self.model, self.data_columns)
        self.assertEqual(len(coefs), 241)
        self.assertGreater(coefs[0]["premium_lakhs"], coefs[-1]["premium_lakhs"])

    def test_sqft_and_bhk_parsers(self) -> None:
        self.assertEqual(parse_sqft_value("2100 - 2850"), 2475.0)
        self.assertEqual(parse_sqft_value("1250"), 1250.0)
        self.assertIsNone(parse_sqft_value("34.46Sq. Meter"))
        self.assertEqual(parse_bhk_value("3 BHK"), 3)
        self.assertEqual(parse_bhk_value("4 Bedroom"), 4)

    def test_market_benchmarks(self) -> None:
        city = get_city_benchmarks(self.market_df)
        self.assertGreater(city["listing_count"], 12000)
        self.assertGreater(city["median_ppsqft"], 3000)

        whitefield = get_locality_benchmark(self.market_df, "whitefield")
        self.assertIsNotNone(whitefield)
        self.assertGreater(whitefield["listing_count"], 100)

        top = get_top_localities_summary(self.market_df, min_listings=25, top_n=10)
        self.assertEqual(len(top), 10)


if __name__ == "__main__":
    unittest.main()
