GRADE_HIERARCHY = ['a', 'b', 'c', 'd', 'e']
HARMFUL_ATTRIBUTES = ['energy100g', 'saturatedfat100g', 'sugars100g', 'sodium100g']
ATTRIBUTES = ['fiber100g', 'proteins100g']
ATTRIBUTES.extend(HARMFUL_ATTRIBUTES)
PETALES = True

WEIGHTS = {'energy100g': 1, 'saturatedfat100g': 1, 'sugars100g': 1, 'sodium100g': 1, 'fiber100g': 2, 'proteins100g': 2}
