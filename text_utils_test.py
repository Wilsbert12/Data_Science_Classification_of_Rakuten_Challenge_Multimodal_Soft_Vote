import text_utils
import sys
import math
def test_clean_description():
    # perfect match between both strings
    str1 = 'Ich habe einen String'
    str2 = 'ICH HABE EINEN STRING'
    assert math.isnan(text_utils.clean_description(str1, str2))
    # string 1 is longer than string 2
    str1 = 'Ich habe einen String der lang ist'
    str2 = 'ICH HABE EINEN STRING'
    assert math.isnan(text_utils.clean_description(str1, str2))
    # string 2 is longer than string 1
    str1 = 'Ich habe einen String'
    str2 = 'ICH HABE EINEN STRING DER LANG IST'
    assert text_utils.clean_description(str1, str2) == 'ICH HABE EINEN STRING DER LANG IST'
    # not a good match between both strings
    str1 = 'ich habe keinen match hier'
    str2 = 'ICH HABE EINEN STRING'
    assert text_utils.clean_description(str1, str2) == 'ICH HABE EINEN STRING'