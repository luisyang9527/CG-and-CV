s = "abcdeafabczikopk"
dic = {}
for i in range(len(s)):
    if s[i] not in dic.keys():
        dic[s[i]] = [s.find(s[i]), s.rfind(s[i])]
# for key in dic.keys():











