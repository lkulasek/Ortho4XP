"""Test script to verify tile directory naming logic"""

def get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix):
    """Calculate tile directory name in format +XX+XXX or -XX-XXX."""
    # Apply sign based on N/S and E/W first
    if lat_prefix == 'S':
        lat_num = -lat_num
    if lon_prefix == 'W':
        lon_num = -lon_num

    # Round down to nearest 10 for directory grouping (floor division)
    # For negative numbers, we want to round towards more negative
    lat_dir = (lat_num // 10) * 10
    lon_dir = (lon_num // 10) * 10

    # Format as +XX+XXX or -XX-XXX
    lat_sign = '+' if lat_dir >= 0 else '-'
    lon_sign = '+' if lon_dir >= 0 else '-'

    return f"{lat_sign}{abs(lat_dir):02d}{lon_sign}{abs(lon_dir):03d}"

# Test cases
test_cases = [
    (54, 26, 'N', 'E', '+50+020'),  # N54E26 -> +50+020
    (50, 10, 'N', 'E', '+50+010'),  # N50E10 -> +50+010
    (50, 0, 'N', 'E', '+50+000'),   # N50E00 -> +50+000
    (60, 20, 'N', 'E', '+60+020'),  # N60E20 -> +60+020
    (40, 30, 'N', 'E', '+40+030'),  # N40E30 -> +40+030
    (36, 121, 'N', 'W', '+30-130'), # N36W121 -> +30-130
    (5, 15, 'S', 'E', '-10+010'),   # S05E15 -> -10+010
    (25, 115, 'S', 'W', '-30-120'), # S25W115 -> -30-120
]

print("Testing tile directory naming:\n")
all_passed = True
for lat_num, lon_num, lat_prefix, lon_prefix, expected in test_cases:
    result = get_tile_directory(lat_num, lon_num, lat_prefix, lon_prefix)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_passed = False
    print(f"{status} {lat_prefix}{lat_num:02d}{lon_prefix}{lon_num:03d} -> {result} (expected: {expected})")

if all_passed:
    print("\n✓ All tests passed!")
else:
    print("\n✗ Some tests failed!")

