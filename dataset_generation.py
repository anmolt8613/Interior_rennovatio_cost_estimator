# Improved synthetic dataset generator for India-wide renovation costs (city-wise granularity)
# The code will generate a large dataset and save it as a CSV you can download.
# You can re-run with different N to scale up as needed.

import random
import math
import pandas as pd
import numpy as np
from datetime import date

random.seed(42)
np.random.seed(42)

# -----------------------------
# Configurable parameters
# -----------------------------
N_ROWS = 10000  # scalable; adjust if you want more/less
AS_OF_DATE = "2025-09-07"
INCLUDE_GST = True
GST_RATE = 0.18  # effective GST on overall contract (approx; mixed supplies/services)

# Monthly material price index (1.0 = base). If you want to up-index later months, change this.
MATERIAL_PRICE_INDEX = 1.00

# -----------------------------
# Master data: Cities (tier, regional multiplier, typical daily labor rate)
# Multipliers roughly reflect metro vs tier-2/3 cost pressures.
# Daily labor rate in INR for skilled+semi-skilled blended average.
# -----------------------------
cities = {
    # Metros
    "Delhi":      {"tier": "Metro", "multiplier": 1.20, "labor_day_rate": (750, 900)},
    "Mumbai":     {"tier": "Metro", "multiplier": 1.30, "labor_day_rate": (800, 950)},
    "Bangalore":  {"tier": "Metro", "multiplier": 1.25, "labor_day_rate": (750, 900)},
    "Hyderabad":  {"tier": "Metro", "multiplier": 1.15, "labor_day_rate": (700, 850)},
    "Chennai":    {"tier": "Metro", "multiplier": 1.10, "labor_day_rate": (650, 800)},
    "Kolkata":    {"tier": "Metro", "multiplier": 1.00, "labor_day_rate": (600, 750)},
    "Pune":       {"tier": "Metro", "multiplier": 1.20, "labor_day_rate": (700, 850)},
    "Ahmedabad":  {"tier": "Metro", "multiplier": 1.05, "labor_day_rate": (600, 750)},
    # Tier-2
    "Lucknow":    {"tier": "Tier-2", "multiplier": 0.92, "labor_day_rate": (500, 650)},
    "Jaipur":     {"tier": "Tier-2", "multiplier": 0.95, "labor_day_rate": (550, 700)},
    "Indore":     {"tier": "Tier-2", "multiplier": 0.96, "labor_day_rate": (550, 700)},
    "Surat":      {"tier": "Tier-2", "multiplier": 1.00, "labor_day_rate": (550, 700)},
    "Nagpur":     {"tier": "Tier-2", "multiplier": 0.95, "labor_day_rate": (520, 680)},
    "Vadodara":   {"tier": "Tier-2", "multiplier": 0.98, "labor_day_rate": (550, 700)},
    "Coimbatore": {"tier": "Tier-2", "multiplier": 0.95, "labor_day_rate": (520, 660)},
    "Bhopal":     {"tier": "Tier-2", "multiplier": 0.93, "labor_day_rate": (500, 640)},
    "Chandigarh": {"tier": "Tier-2", "multiplier": 1.02, "labor_day_rate": (600, 750)},
    # Tier-3 (representative)
    "Patna":      {"tier": "Tier-3", "multiplier": 0.88, "labor_day_rate": (450, 600)},
    "Ranchi":     {"tier": "Tier-3", "multiplier": 0.90, "labor_day_rate": (480, 620)},
    "Guwahati":   {"tier": "Tier-3", "multiplier": 0.92, "labor_day_rate": (500, 640)},
    "Kanpur":     {"tier": "Tier-3", "multiplier": 0.90, "labor_day_rate": (480, 620)},
    "Varanasi":   {"tier": "Tier-3", "multiplier": 0.90, "labor_day_rate": (480, 620)},
    "Mysuru":     {"tier": "Tier-3", "multiplier": 0.94, "labor_day_rate": (520, 660)},
}

city_names = list(cities.keys())

# -----------------------------
# Room types & renovation scope presets
# -----------------------------
room_types = ["Bedroom", "Living Room", "Kitchen", "Bathroom", "Dining", "Study", "Kids Room"]

renovation_levels = {
    "Basic":    {"labor_overhead": (0.10, 0.15)},
    "Mid":      {"labor_overhead": (0.15, 0.25)},
    "Luxury":   {"labor_overhead": (0.25, 0.35)},
}

# Material quality tiers influencing unit prices (brand/finish effect)
quality_tiers = {
    "Economy": 0.90,
    "Standard": 1.00,
    "Premium": 1.20,
    "Luxury":  1.40
}

# -----------------------------
# Component base cost tables (pre-multiplier, pre-quality)
# Values are in INR. Where relevant, costs are per sq.ft. Otherwise per unit space.
# Ranges reflect 2025 ballparks; final price = base * quality * city_multiplier * index
# -----------------------------

# Painting (interior) per sq.ft of built-up area
painting_psf = (18, 35)

# Flooring types per sq.ft (material + install)
flooring_catalog = {
    "Vitrified_Tile": (90, 200),
    "Ceramic_Tile":   (80, 160),
    "Wood_Laminate":  (140, 260),
    "Engineered_Wood":(220, 380),
    "Marble":         (350, 900),   # highly variant by marble type
    "Granite":        (220, 450),
    "Epoxy":          (60, 160)
}

# False ceiling per sq.ft
ceiling_catalog = {
    "POP":    (90, 160),
    "Gypsum": (120, 220),
    "Grid":   (100, 180),
}

# Electrical re-wiring per sq.ft
electrical_psf = (70, 160)

# Plumbing refresh (usually per wet area unit, but we also support per bathroom/kitchen scope flags)
plumbing_per_bath = (40000, 95000)
plumbing_per_kitchen = (25000, 60000)

# Modular kitchen packages (per kitchen)
kitchen_packages = {
    "Basic_L":       (120000, 250000),
    "Standard_L":    (180000, 350000),
    "Premium_U":     (300000, 600000),
    "Luxury_Island": (500000, 1200000),
}

# Bathroom packages (per bathroom)
bathroom_packages = {
    "Basic":   (70000, 130000),
    "Standard":(120000, 220000),
    "Premium": (200000, 400000),
}

# Furniture/storages per room (optional, broad range)
furniture_room = {
    "None":     (0, 0),
    "Basic":    (30000, 80000),
    "Standard": (80000, 180000),
    "Premium":  (180000, 500000)
}

# Productivity assumptions for labor conversion (very rough; used to compute labor_psf from day rates)
# e.g., a crew can cover 180-280 sq.ft/day for painting prep+2 coats in practice (varies widely).
productivity = {
    "painting_sqft_per_day": (180, 280),
    "flooring_sqft_per_day": (90, 160),
    "ceiling_sqft_per_day":  (120, 180),
    "electrical_sqft_per_day": (150, 240),
}

# -----------------------------
# Helper functions
# -----------------------------
def rand_range(a, b):
    return random.uniform(a, b)

def pick_weighted(d):
    # d is dict of {key: weight}
    keys = list(d.keys())
    weights = list(d.values())
    return random.choices(keys, weights=weights, k=1)[0]

def choose_quality():
    return random.choices(list(quality_tiers.keys()), weights=[0.25, 0.45, 0.22, 0.08], k=1)[0]

def choose_flooring_type(room_type):
    # kitchens & bathrooms often have tile/stone; bedrooms/living can be wider
    if room_type in ["Kitchen", "Bathroom"]:
        choices = ["Ceramic_Tile", "Vitrified_Tile", "Granite"]
        weights = [0.45, 0.40, 0.15]
    else:
        choices = list(flooring_catalog.keys())
        weights = [0.25, 0.35, 0.10, 0.08, 0.10, 0.07, 0.05]  # bias to vitrified/ceramic
    return random.choices(choices, weights=weights, k=1)[0]

def choose_ceiling_type():
    return random.choice(list(ceiling_catalog.keys()))

def choose_furniture_level(room_type):
    if room_type in ["Kitchen", "Bathroom"]:
        return "None"
    return random.choices(["None", "Basic", "Standard", "Premium"], weights=[0.35, 0.35, 0.22, 0.08], k=1)[0]

def kitchen_package_by_level(renov_level):
    if renov_level == "Basic":
        return "Basic_L"
    if renov_level == "Mid":
        return random.choice(["Basic_L", "Standard_L"])
    return random.choice(["Standard_L", "Premium_U", "Luxury_Island"])

def bathroom_package_by_level(renov_level):
    if renov_level == "Basic":
        return "Basic"
    if renov_level == "Mid":
        return random.choice(["Basic", "Standard"])
    return random.choice(["Standard", "Premium"])

def psf_cost(base_rng, quality_factor, city_mult):
    base = rand_range(*base_rng)
    return base * quality_factor * city_mult * MATERIAL_PRICE_INDEX

def unit_cost(base_rng, quality_factor, city_mult):
    base = rand_range(*base_rng)
    return base * quality_factor * city_mult * MATERIAL_PRICE_INDEX

def labor_component_cost(area_sqft, city_info, category):
    # approximate labor cost for category based on daily rate & productivity
    day_rate = rand_range(*city_info["labor_day_rate"])
    if category == "painting":
        sqft_per_day = rand_range(*productivity["painting_sqft_per_day"])
    elif category == "flooring":
        sqft_per_day = rand_range(*productivity["flooring_sqft_per_day"])
    elif category == "ceiling":
        sqft_per_day = rand_range(*productivity["ceiling_sqft_per_day"])
    elif category == "electrical":
        sqft_per_day = rand_range(*productivity["electrical_sqft_per_day"])
    else:
        sqft_per_day = 150.0
    days = max(1.0, area_sqft / sqft_per_day)
    # Assume small crew encoded in productivity; day_rate approximates per-person blend
    return day_rate * days

# -----------------------------
# Row generator
# -----------------------------
def generate_row(idx):
    city = random.choice(city_names)
    cinfo = cities[city]
    tier = cinfo["tier"]
    city_mult = cinfo["multiplier"]
    room_type = random.choice(room_types)
    area = int(rand_range(70, 550))  # sq.ft for a single room
    renov_level = random.choice(list(renovation_levels.keys()))
    labor_overhead_pct = round(rand_range(*renovation_levels[renov_level]["labor_overhead"]), 3)

    # Quality selections
    paint_quality = choose_quality()
    floor_quality = choose_quality()
    ceiling_quality = choose_quality()

    # Painting
    painting_rate_psf = psf_cost(painting_psf, quality_tiers[paint_quality], city_mult)
    painting_material_cost = painting_rate_psf * area
    painting_labor_cost = labor_component_cost(area, cinfo, "painting")

    # Flooring
    floor_type = choose_flooring_type(room_type)
    floor_rate_psf = psf_cost(flooring_catalog[floor_type], quality_tiers[floor_quality], city_mult)
    flooring_material_cost = floor_rate_psf * area
    flooring_labor_cost = labor_component_cost(area, cinfo, "flooring")

    # False ceiling (50–80% rooms get it except bathrooms typically)
    has_ceiling = (room_type != "Bathroom") and (random.random() < 0.65)
    ceiling_type = choose_ceiling_type() if has_ceiling else "None"
    ceiling_area = int(area * rand_range(0.8, 1.05)) if has_ceiling else 0
    ceiling_rate_psf = psf_cost(ceiling_catalog[ceiling_type], quality_tiers[ceiling_quality], city_mult) if has_ceiling else 0.0
    ceiling_material_cost = ceiling_rate_psf * ceiling_area
    ceiling_labor_cost = labor_component_cost(ceiling_area, cinfo, "ceiling") if has_ceiling else 0.0

    # Electrical rewiring (40–90% cases depending on level)
    has_electrical = random.random() < {"Basic":0.4, "Mid":0.65, "Luxury":0.9}[renov_level]
    electrical_rate_psf = psf_cost(electrical_psf, 1.0, city_mult) if has_electrical else 0.0
    electrical_material_cost = electrical_rate_psf * area
    electrical_labor_cost = labor_component_cost(area, cinfo, "electrical") if has_electrical else 0.0

    # Kitchen package if room is Kitchen
    kitchen_pkg_name, kitchen_pkg_cost = "None", 0.0
    if room_type == "Kitchen":
        kname = kitchen_package_by_level(renov_level)
        kcost = unit_cost(kitchen_packages[kname], quality_tiers[choose_quality()], city_mult)
        kitchen_pkg_name, kitchen_pkg_cost = kname, kcost

    # Bathroom package if room is Bathroom
    bath_pkg_name, bath_pkg_cost = "None", 0.0
    if room_type == "Bathroom":
        bname = bathroom_package_by_level(renov_level)
        bcost = unit_cost(bathroom_packages[bname], quality_tiers[choose_quality()], city_mult)
        bath_pkg_name, bath_pkg_cost = bname, bcost

    # Plumbing add-ons for Kitchen/Bathroom
    plumbing_cost = 0.0
    if room_type == "Kitchen":
        plumbing_cost += unit_cost(plumbing_per_kitchen, 1.0, city_mult)
    if room_type == "Bathroom":
        plumbing_cost += unit_cost(plumbing_per_bath, 1.0, city_mult)

    # Furniture/Storage (not for Kitchen/Bathroom in this simple model)
    furniture_level = choose_furniture_level(room_type)
    furniture_cost = 0.0
    if furniture_level != "None":
        furniture_cost = unit_cost(furniture_room[furniture_level], 1.0, city_mult)

    # Wastage & sundries (5–10% of materials)
    material_subtotal = (
        painting_material_cost + flooring_material_cost + ceiling_material_cost +
        electrical_material_cost + kitchen_pkg_cost + bath_pkg_cost + plumbing_cost + furniture_cost
    )
    wastage_pct = rand_range(0.05, 0.10)
    wastage_cost = material_subtotal * wastage_pct

    # Labor subtotal (explicit categories) + contractor overhead
    labor_subtotal = painting_labor_cost + flooring_labor_cost + ceiling_labor_cost + electrical_labor_cost
    contractor_overhead = (material_subtotal + labor_subtotal) * labor_overhead_pct

    # Pre-tax subtotal
    subtotal_pre_tax = material_subtotal + wastage_cost + labor_subtotal + contractor_overhead

    # Apply GST if chosen
    gst_amount = subtotal_pre_tax * GST_RATE if INCLUDE_GST else 0.0
    grand_total = subtotal_pre_tax + gst_amount

    # Per sq.ft sanity metric
    total_psf = grand_total / max(1, area)

    return {
        "Row_ID": idx,
        "As_Of_Date": AS_OF_DATE,
        "Material_Price_Index": MATERIAL_PRICE_INDEX,
        "City": city,
        "City_Tier": tier,
        "City_Multiplier": city_mult,
        "Labor_Day_Rate_Min": cities[city]["labor_day_rate"][0],
        "Labor_Day_Rate_Max": cities[city]["labor_day_rate"][1],
        "Room_Type": room_type,
        "Area_Sqft": area,
        "Renovation_Level": renov_level,
        "Paint_Quality": paint_quality,
        "Floor_Type": floor_type,
        "Floor_Quality": floor_quality,
        "Ceiling_Type": ceiling_type,
        "Ceiling_Quality": ceiling_quality if has_ceiling else "None",
        "Has_Electrical": has_electrical,
        "Furniture_Level": furniture_level,
        "Kitchen_Package": kitchen_pkg_name,
        "Bathroom_Package": bath_pkg_name,
        # Cost components (rounded to nearest rupee)
        "Painting_Material_Cost": round(painting_material_cost),
        "Painting_Labor_Cost": round(painting_labor_cost),
        "Flooring_Material_Cost": round(flooring_material_cost),
        "Flooring_Labor_Cost": round(flooring_labor_cost),
        "Ceiling_Material_Cost": round(ceiling_material_cost),
        "Ceiling_Labor_Cost": round(ceiling_labor_cost),
        "Electrical_Material_Cost": round(electrical_material_cost),
        "Electrical_Labor_Cost": round(electrical_labor_cost),
        "Kitchen_Package_Cost": round(kitchen_pkg_cost),
        "Bathroom_Package_Cost": round(bath_pkg_cost),
        "Plumbing_Cost": round(plumbing_cost),
        "Furniture_Cost": round(furniture_cost),
        "Wastage_Sundries_Cost": round(wastage_cost),
        "Contractor_Overhead_Cost": round(contractor_overhead),
        "GST_Amount": round(gst_amount),
        "Grand_Total": round(grand_total),
        "Total_Cost_per_Sqft": round(total_psf, 2)
    }

# -----------------------------
# Generate dataset
# -----------------------------
rows = [generate_row(i) for i in range(1, N_ROWS + 1)]
df = pd.DataFrame(rows)

# Save to CSV and preview head/tail to user
csv_path = "/mnt/data/renovation_cost_dataset_india_v2.csv"
df.to_csv(csv_path, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("India Renovation Cost Dataset v2 (10k rows, city-wise) — Preview", df.head(50))

csv_path
