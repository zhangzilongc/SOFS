VISION_V1_split_train_dict = {
    "split_0": ['Console', 'Casting', 'Groove', 'Capacitor', 'PCB_2', 'Screw', 'Cylinder', 'Electronics', 'Lens'],
    "split_1": ['PCB_1', 'Groove', 'Cable', 'Hemisphere', 'Screw', 'Cylinder', 'Wood', 'Ring', 'Lens'],
    "split_2": ['Console', 'PCB_1', 'Casting', 'Cable', 'Hemisphere', 'Capacitor', 'PCB_2', 'Wood', 'Electronics',
                'Ring'],
}

# VISION_V1_split_train_dict = {
#     "split_0": ['Console', 'Casting', 'Groove', 'Capacitor', 'PCB_2', 'Screw', 'Cylinder', 'Electronics', 'Lens',
#                 "PCB_1", "Hemisphere", "Wood", "Ring", "Cable"]
# }

VISION_V1_split_train_val_dict = {
    "split_0": ['Console', 'Casting', 'Groove', 'Capacitor', 'PCB_2', 'Screw', 'Cylinder', 'Electronics', 'Lens'],
    "split_1": ['PCB_1', 'Groove', 'Cable', 'Hemisphere', 'Screw', 'Cylinder', 'Wood', 'Ring', 'Lens'],
    "split_2": ['Console', 'PCB_1', 'Casting', 'Cable', 'Hemisphere', 'Capacitor', 'PCB_2', 'Wood', 'Electronics',
                'Ring'],
}

VISION_V1_split_test_dict = {
    "split_0": ["PCB_1", "Hemisphere", "Wood", "Ring", "Cable"],
    "split_1": ["Casting", "Capacitor", "PCB_2", "Electronics", "Console"],
    "split_2": ["Groove", "Lens", "Screw", "Cylinder"],
}

DS_Spectrum_DS_split_test_dict = {
    "split_0": [
        "pill", "leather", "carpet", "hazelnut", "transistor",
        "DS-DAGM", "capsule", "screw", "wood", "zipper", "DS-Cotton-Fabric",
        "toothbrush", "grid", "cable", "metal_nut", "tile", "bottle"
    ]
}

DS_Spectrum_split_train_dict = {
    "split_0": ["Wood_VISION", "Console_VISION", "pill", "leather", "carpet", "hazelnut", "transistor",
                "Groove_VISION", "DS-DAGM", "capsule", "screw", "wood", "zipper", "DS-Cotton-Fabric", "bottle"],
    "split_1": ["Screw_VISION", "Ring_VISION", "Capacitor_VISION", "toothbrush", "grid", "cable", "metal_nut", "tile",
                "Groove_VISION", "DS-DAGM", "capsule", "screw", "wood", "zipper", "DS-Cotton-Fabric", "bottle"],
    "split_2": ["Screw_VISION", "Ring_VISION", "Capacitor_VISION", "toothbrush", "grid", "cable", "metal_nut", "tile",
                "Wood_VISION", "Console_VISION", "pill", "leather", "carpet", "hazelnut", "transistor"],
}

DS_Spectrum_split_test_dict = {
    "split_0": ["Screw_VISION", "Ring_VISION", "Capacitor_VISION", "toothbrush", "grid", "cable", "metal_nut", "tile"],
    "split_1": ["Wood_VISION", "Console_VISION", "pill", "leather", "carpet", "hazelnut", "transistor"],
    "split_2": ["Groove_VISION", "DS-DAGM", "capsule", "screw", "wood", "zipper", "DS-Cotton-Fabric", "bottle"],
}
