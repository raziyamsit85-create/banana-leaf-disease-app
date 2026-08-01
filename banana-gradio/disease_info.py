"""Human-readable info shown next to each prediction."""

DISEASE_INFO = {
    "Banana Healthy Leaf": {
        "bn": "সুস্থ কলা পাতা",
        "severity": "healthy",
        "summary": "No disease symptoms detected. Leaf tissue appears uniform and healthy.",
        "advice": [
            "Maintain current irrigation and nutrition schedule.",
            "Keep monitoring weekly, especially during humid periods.",
            "Remove old or fallen leaves to reduce inoculum build-up.",
        ],
    },
    "Banana Insect Pest Disease": {
        "bn": "পোকামাকড়ের আক্রমণ",
        "severity": "moderate",
        "summary": "Feeding damage consistent with insect pests such as banana aphid, thrips or leaf beetle.",
        "advice": [
            "Identify the pest before spraying; scout the underside of leaves.",
            "Consider neem-based or recommended IPM treatment.",
            "Aphids also transmit Bunchy Top virus, so control them early.",
        ],
    },
    "Banana Moko Disease or dead": {
        "bn": "মোকো রোগ",
        "severity": "severe",
        "summary": "Bacterial wilt caused by Ralstonia solanacearum race 2. Highly destructive and spreads via tools, soil and insects.",
        "advice": [
            "Isolate and destroy infected mats; do not compost them.",
            "Disinfect all cutting tools between plants.",
            "Fallow or rotate the affected block; there is no curative spray.",
        ],
    },
    "Banana Yellow Sigatoka Disease": {
        "bn": "ইয়েলো সিগাটোকা",
        "severity": "moderate",
        "summary": "Leaf spot caused by Mycosphaerella musicola. Reduces photosynthetic area and delays bunch filling.",
        "advice": [
            "Deleaf and destroy heavily spotted leaves.",
            "Improve drainage and plant spacing to lower canopy humidity.",
            "Apply a recommended protectant fungicide on a scouting-based schedule.",
        ],
    },
    "Black Sigotika": {
        "bn": "ব্ল্যাক সিগাটোকা",
        "severity": "severe",
        "summary": "Black Sigatoka (Mycosphaerella fijiensis). More aggressive than the yellow form and can cause major yield loss.",
        "advice": [
            "Start control early; the disease progresses fast in warm, wet weather.",
            "Rotate fungicide modes of action to avoid resistance.",
            "Remove and bury infected leaf material away from the field.",
        ],
    },
    "Fusarium Wilt Panama": {
        "bn": "পানামা রোগ / ফিউজেরিয়াম উইল্ট",
        "severity": "severe",
        "summary": "Soil-borne fungal wilt (Fusarium oxysporum f. sp. cubense). The pathogen persists in soil for decades.",
        "advice": [
            "Quarantine the affected area and restrict movement of soil and water.",
            "Use certified disease-free planting material only.",
            "Plan for resistant cultivars; chemical control is not effective.",
        ],
    },
}


def get_info(class_name):
    return DISEASE_INFO.get(
        class_name,
        {
            "bn": "",
            "severity": "unknown",
            "summary": "No additional information available.",
            "advice": [],
        },
    )
