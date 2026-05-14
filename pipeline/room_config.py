"""room_config.py — room-type-specific category lists + phase 2 prompt
extras. Drives inventory.py (which categories per pass) and
_phase2_detect.py (which extras to inject into KEEP / SKIP sections).

Room types:
  living_room, dining_room, kitchen, bedroom, office, bathroom,
  hallway, mixed (open-plan / combined)

If a scene is detected as "mixed" or detection is uncertain, the mixed
config covers the union so we don't miss anything.
"""

ROOM_CONFIGS = {
    "living_room": {
        "inventory_passes": [
            ("pass0_shelving", ["bookshelf", "bookcase", "open shelving unit",
                                 "etagere", "display shelf", "tall shelving unit"]),
            ("pass1_seating", ["sofa", "armchair", "chair", "ottoman", "stool",
                                "loveseat", "recliner", "bench"]),
            ("pass2_storage", ["cabinet", "sideboard", "credenza", "tv stand",
                                "media console"]),
            ("pass3_tables",  ["coffee table", "side table", "console table",
                                "end table"]),
        ],
        "phase2_keep_extras": [
            "floor-standing lamps", "floor-standing potted plants",
            "TVs / monitors / mirrors mounted on walls or sitting on furniture",
            "free-standing speakers / hi-fi cabinets / record players on the floor",
        ],
        "phase2_skip_extras": [],
    },

    "dining_room": {
        "inventory_passes": [
            ("pass0_shelving", ["bookshelf", "bookcase", "open shelving unit",
                                 "etagere", "display shelf"]),
            ("pass1_seating", ["dining chair", "armchair", "bench"]),
            ("pass2_storage", ["sideboard", "buffet", "cabinet", "credenza",
                                "china cabinet", "hutch"]),
            ("pass3_tables",  ["dining table", "side table", "console table"]),
        ],
        "phase2_keep_extras": [
            "floor-standing lamps", "floor-standing potted plants",
            "mirrors mounted on walls",
        ],
        "phase2_skip_extras": [],
    },

    "kitchen": {
        "inventory_passes": [
            ("pass1_seating", ["bar stool", "kitchen stool", "chair"]),
            ("pass2_islands", ["kitchen island", "kitchen cart",
                                "freestanding cabinet"]),
            ("pass3_tables",  ["kitchen table", "breakfast table"]),
        ],
        "phase2_keep_extras": [
            "free-standing stoves / ovens / range hoods",
            "free-standing refrigerators",
            "kitchen appliances (kettles, coffee makers, microwaves, "
            "toasters, blenders, mixers)",
            "floor-standing potted plants",
        ],
        "phase2_skip_extras": [
            "kitchen base cabinets, kitchen wall cabinets, kitchen built-ins, "
            "and any cabinetry that runs along the kitchen wall",
        ],
    },

    "bedroom": {
        "inventory_passes": [
            ("pass1_bed",     ["bed", "platform bed", "daybed", "bunk bed"]),
            ("pass2_seating", ["chair", "armchair", "stool", "bench",
                                "ottoman"]),
            ("pass3_storage", ["dresser", "wardrobe", "armoire",
                                "chest of drawers", "nightstand", "bedside table",
                                "vanity"]),
            ("pass4_tables",  ["desk", "side table", "writing desk"]),
        ],
        "phase2_keep_extras": [
            "bookshelves / bookcases / open shelving units",
            "floor-standing lamps", "floor-standing potted plants",
            "TVs / monitors / mirrors mounted on walls or sitting on furniture",
            "freestanding wardrobes / armoires / chests not caught by topdown",
        ],
        "phase2_skip_extras": [
            "built-in closets, walk-in closet doors, recessed wardrobes",
            "items lying on the bed (pillows, blankets, throws — these are "
            "part of the bed's bbox)",
        ],
    },

    "office": {
        "inventory_passes": [
            ("pass1_desks",    ["desk", "writing desk", "executive desk",
                                 "credenza", "conference table"]),
            ("pass2_seating",  ["office chair", "desk chair", "task chair",
                                 "armchair", "couch", "bench"]),
            ("pass3_storage",  ["filing cabinet", "bookshelf", "bookcase",
                                 "credenza", "cabinet", "lateral file"]),
            ("pass4_tables",   ["side table", "coffee table", "console table"]),
        ],
        "phase2_keep_extras": [
            "tall bookshelves / bookcases / open shelving units / filing "
            "towers (treat each unit as ONE object)",
            "floor-standing lamps", "floor-standing potted plants",
            "monitors / TVs sitting on desks or mounted on walls",
            "printers / multi-function printers / paper shredders",
            "free-standing whiteboards / cork boards / easels",
        ],
        "phase2_skip_extras": [
            "items sitting on the desk (keyboards, mice, papers, mugs, "
            "pens, books — these belong to the desk's bbox)",
        ],
    },

    "bathroom": {
        "inventory_passes": [
            ("pass1_fixtures", ["toilet", "freestanding bathtub", "vanity",
                                 "freestanding shower"]),
            ("pass2_storage",  ["cabinet", "linen cabinet", "vanity cabinet"]),
            ("pass3_seating",  ["stool", "bench", "chair"]),
        ],
        "phase2_keep_extras": [
            "floor-standing potted plants",
            "mirrors mounted on walls",
            "wall-mounted shelves / open shelving",
            "towel racks (free-standing)",
        ],
        "phase2_skip_extras": [
            "wall-mounted toiletry holders, soap dishes, shower fixtures, "
            "toilet paper holders, towel bars",
        ],
    },

    "hallway": {
        "inventory_passes": [
            ("pass1_seating", ["bench", "stool", "chair"]),
            ("pass2_storage", ["console table", "credenza", "shoe cabinet",
                                "coat stand", "umbrella stand"]),
            ("pass3_tables",  ["console table", "side table"]),
        ],
        "phase2_keep_extras": [
            "tall narrow bookshelves / shelving units",
            "floor-standing lamps", "floor-standing potted plants",
            "mirrors mounted on walls",
        ],
        "phase2_skip_extras": [],
    },

    # Catch-all when scene combines multiple room types (open-plan
    # kitchen+living+dining is the canonical example) OR when room
    # detection returned uncertain. UNION of all categories so nothing
    # is missed.
    "mixed": {
        "inventory_passes": [
            ("pass0_shelving", ["bookshelf", "bookcase", "open shelving unit",
                                 "etagere", "display shelf", "filing tower"]),
            ("pass1_seating",  ["sofa", "armchair", "chair", "ottoman",
                                 "stool", "loveseat", "recliner", "bench",
                                 "dining chair", "office chair", "bar stool"]),
            ("pass2_bed",      ["bed", "platform bed", "daybed"]),
            ("pass3_storage",  ["cabinet", "sideboard", "credenza",
                                 "tv stand", "media console", "dresser",
                                 "wardrobe", "chest of drawers", "nightstand",
                                 "filing cabinet"]),
            ("pass4_tables",   ["coffee table", "dining table", "side table",
                                 "console table", "desk", "kitchen table"]),
            ("pass5_kitchen",  ["kitchen island", "kitchen cart"]),
        ],
        "phase2_keep_extras": [
            "floor-standing lamps", "floor-standing potted plants",
            "TVs / monitors / mirrors mounted on walls or sitting on furniture",
            "free-standing stoves / ovens / range hoods",
            "free-standing refrigerators",
            "kitchen appliances (kettles, coffee makers, microwaves, "
            "toasters, blenders, mixers)",
        ],
        "phase2_skip_extras": [
            "kitchen base cabinets, kitchen wall cabinets, kitchen built-ins, "
            "any cabinetry running along the kitchen wall",
        ],
    },
}


VALID_ROOM_TYPES = list(ROOM_CONFIGS.keys())


def get_room_config(room_type: str) -> dict:
    """Look up a room config; fall back to mixed if unknown."""
    rt = (room_type or "").lower().strip().replace(" ", "_").replace("-", "_")
    return ROOM_CONFIGS.get(rt, ROOM_CONFIGS["mixed"])
