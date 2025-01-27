import re
from sympy import simplify
from sympy.parsing.latex import parse_latex

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_multiple_slashes(string):
    # template: for all int a/ int b --> \frac{a}{b}
    matchs = re.findall(r"\d+/\d+", string)
    for match in matchs:
        string = string.replace(match, _fix_a_slash_b(match))
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    elif "\mbox{" in string:
        splits = string.split("\mbox{")
        assert len(splits) == 2
        return splits[0]
    else:
        return string
    
def _remove_all_text(string):
    ## \text{A} --> A
    ## \text{(A)} --> A
    if "\\text{" not in string:
        return string
    if string.startswith("\\text{(") and string.endswith(")}"):
        return string[7:-2]
    if string.startswith("\\text{") and string.endswith("}"):
        return string[6:-1]


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_point5(string):
    # a.5 --> \frac{2a+1}{2}
    if ".5" not in string:
        return string
    matchs = re.findall(r"\d+.5", string)
    # 0.50 -> 0.5
    if "0.50" in string:
        string = string.replace("0.50", "0.5")
    for match in matchs:
        a = match[:-2]
        new_string = "\\frac{" + str(2*int(a) + 1) + "}{2}"
        string = string.replace(match, new_string)
    return string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # consider x \in 
    if len(string.split("\\in")) == 2:
        if len(string.split("\\in")[0]) <= 2 and (not string.split("\\in")[1].startswith("fty")):
        # if len(string.split("\\in")[0]) <= 2:
            string = string.split("\\in")[1]

    # remove spaces
    string = string.replace(" ", "")

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove all text
    string = _remove_all_text(string)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    string = _fix_point5(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    # string = _fix_a_slash_b(string)
    string = _fix_multiple_slashes(string)

    return string

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def is_equiv_old(str1, str2, verbose=False):
    str1 = last_boxed_only_string(str1)
    str2 = last_boxed_only_string(str2)
    str1 = remove_boxed(str1)
    str2 = remove_boxed(str2)
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2
    


def is_equiv(str1, str2, verbose=False):
    str1 = last_boxed_only_string(str1)
    str2 = last_boxed_only_string(str2)
    str1 = remove_boxed(str1)
    str2 = remove_boxed(str2)
    
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        print("WARNING: One None")
        return False
    try:
        str1 = _strip_string(str1)
        str2 = _strip_string(str2)
        s1 = parse_latex(str1)
        s2 = parse_latex(str2)
        equal = s1.equals(s2)
        if not equal:
            s1 = simplify(s1)
            s2 = simplify(s2)
            tol = 1e-6
            equal = abs(s1.evalf() - s2.evalf()) < tol
        # assert is bool
        assert type(equal) == bool
        return equal
    except:
        return str1 == str2
    

if __name__ == "__main__":
    # test cases
    #  864\mbox{inches}^2 
    # \frac{1}{2}0
    # \infty 
    print(is_equiv("\\boxed{[\infty,0)}", "\\boxed{0.50}"))
    print(is_equiv("\\boxed{0.5}", "\\boxed{\\frac{1}{2}}"))