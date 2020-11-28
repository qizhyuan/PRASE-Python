
class PARISConfig:
    def __init__(self, theta=0.1, similarity_fuc=None):
        self._similarity_fuc = similarity_fuc
        self.theta = theta

    def get_similarity(self, s1: str, s2: str):
        if self._similarity_fuc is not None:
            return self._similarity_fuc(s1, s2)
        Levenshtein_distance = 0
        s_short, s_long = s1, s2
        if len(s_short) > len(s_long):
            s_short, s_long = s2, s1
        if len(s_short) == 0:
            Levenshtein_distance = len(s_long)
        else:
            def get_element(matrix, idx1, idx2):
                if idx1 <= 0 and idx2 <= 0:
                    return 0
                if idx1 <= 0:
                    return idx2
                if idx2 <= 0:
                    return idx1
                return matrix[idx1][idx2 % 2]
            dp = [[0] * 2 for _ in range(len(s_short) + 1)]
            for j in range(1, len(s_long) + 1):
                for i in range(1, len(s_short) + 1):
                    dp[i][j % 2] = min(get_element(dp, i, j - 1) + 1, get_element(dp, i - 1, j) + 1,
                                       get_element(dp, i - 1, j - 1) + (0 if s_short[i - 1] == s_long[j - 1] else 1))
            Levenshtein_distance = get_element(dp, len(s_short), len(s_long))
        print(Levenshtein_distance)
        similarity_score = 1.0 - (float(Levenshtein_distance) / max(len(s_short), len(s_long)))
        return similarity_score



