import textdistance
import itertools
import unicodedata

# 1. levenshtein            (Levenshtein)
# 2. damerau_levenshtein    (Damerau-Levenshtein)
# 3. jaro                   (Jaro)
# 4. jaro_winkler           (Jaro-Winkler)
# 5. sorted_jaro_winkler    (Sorted Jaro-Winkler)
# 6. cosine                 (Cosine similarity)
# 7. jaccard                (Jaccard similarity)
# 8. overlap                (Overlap coefficient)
# 9. dice                   (Dice coefficient)
# 10. soft_jaccard          (Soft-Jaccard)
# 11. monge_elkan           (Monge-Elkan)
# 12. permuted_winkler      (Permuted Jaro-Winkler)
# 13. skipgram              (Jaccard Skipgrams)
# 14. davies                (Davis and De Salles)

def levenshtein(str1, str2):
    if (len(str1) >= len(str2)):
        max_len = len(str1)
    else:
        max_len = len(str2)
    return 1.0 - textdistance.levenshtein(str1, str2)/max_len


def damerau_levenshtein(str1, str2):
    if (len(str1) >= len(str2)):
        max_len = len(str1)
    else:
        max_len = len(str2)
    return 1.0 - textdistance.damerau_levenshtein(str1, str2)/max_len


def jaro(str1, str2):
    return textdistance.jaro(str1, str2)


def jaro_winkler(str1, str2):
    return textdistance.jaro_winkler(str1, str2)


def sorted_jaro_winkler(str1, str2):
    a = sorted(str1.split(" "))
    b = sorted(str2.split(" "))
    a = " ".join(a)
    b = " ".join(b)
    return textdistance.jaro_winkler(a, b)


def cosine(str1, str2):
    return textdistance.cosine(str1, str2)


def jaccard(str1, str2):
    return textdistance.jaccard(str1, str2)


def overlap(str1, str2):
    return textdistance.overlap(str1, str2)


def dice(str1, str2):
    return textdistance.sorensen_dice(str1, str2)


def soft_jaccard(str1, str2):
    a = set(str1.split(" "))
    b = set(str2.split(" "))
    intersection_length = (sum(max(jaro_winkler(i, j) for j in b)
                           for i in a) + sum(max(jaro_winkler(i, j) for j in a) for i in b)) / 2.0
    return float(intersection_length)/(len(a) + len(b) - intersection_length)


def monge_elkan_aux(str1, str2):
    cummax = 0
    for ws in str1.split(" "):
        maxscore = 0
        for wt in str2.split(" "):
            maxscore = max(maxscore, jaro_winkler(ws, wt))
        cummax += maxscore
    return cummax / len(str1.split(" "))


def monge_elkan(str1, str2):
    return (monge_elkan_aux(str1, str2) + monge_elkan_aux(str2, str1)) / 2.0


def permuted_winkler(str1, str2):
    a = str1.split(" ")
    b = str2.split(" ")
    if len(a) > 5:
        a = a[0:5] + [u''.join(a[5:])]
    if len(b) > 5:
        b = b[0:5] + [u''.join(b[5:])]
    lastscore = 0.0
    for a in itertools.permutations(a):
        for b in itertools.permutations(b):
            sa = u' '.join(a)
            sb = u' '.join(b)
            score = jaro_winkler(sa, sb)
            if score > lastscore:
                lastscore = score
    return lastscore


def skipgrams(sequence, n, k):
    sequence = " " + sequence + " "
    res = []
    for ngram in {sequence[i:i+n+k] for i in range(len(sequence) - (n + k - 1))}:
        if k == 0:
            res.append(ngram)
        else:
            res.append(ngram[0:1] + ngram[k+1:len(ngram)])
    return res


def skipgram(str1, str2):
    a1 = set(skipgrams(str1, 2, 0))
    a2 = set(skipgrams(str1, 2, 1) + skipgrams(str1, 2, 2))
    b1 = set(skipgrams(str2, 2, 0))
    b2 = set(skipgrams(str2, 2, 1) + skipgrams(str1, 2, 2))
    c1 = a1.intersection(b1)
    c2 = a2.intersection(b2)
    d1 = a1.union(b1)
    d2 = a2.union(b2)
    try:
        return float(len(c1) + len(c2)) / float(len(d1) + len(d2))
    except:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def davies(str1, str2):
    a = strip_accents(str1.lower()).replace(u'-', u' ').split(' ')
    b = strip_accents(str2.lower()).replace(u'-', u' ').split(' ')
    for i in range(len(a)):
        if len(a[i]) > 1 or not(a[i].endswith(u'.')):
            continue
        replacement = len(str2)
        for j in range(len(b)):
            if b[j].startswith(a[i].replace(u'.', '')):
                if len(b[j]) < replacement:
                    a[i] = b[j]
                    replacement = len(b[j])
    for i in range(len(b)):
        if len(b[i]) > 1 or not(b[i].endswith(u'.')):
            continue
        replacement = len(str1)
        for j in range(len(a)):
            if a[j].startswith(b[i].replace(u'.', '')):
                if len(a[j]) < replacement:
                    b[i] = a[j]
                    replacement = len(a[j])
    a = set(a)
    b = set(b)
    aux1 = sorted_jaro_winkler(str1, str2)
    intersection_length = (sum(max(jaro_winkler(i, j) for j in b)
                           for i in a) + sum(max(jaro_winkler(i, j) for j in a) for i in b)) / 2.0
    aux2 = float(intersection_length)/(len(a) + len(b) - intersection_length)
    return (aux1 + aux2) / 2.0
