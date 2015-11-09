import sys, os, re

def getWordAttributes(filename):
	f = open(filename,'r')
	wordAttr = []
	for line in f:
		L = []
		for i in line.strip().split(' '):
			if len(i) > 0:
				L.append(i)
		wordAttr.append(L)
	f.close()
	return wordAttr

def getDistanceTuple(words, count):
	i, j = count, count+1
	wordDist = [attr[0] for attr in wordAttr]
	while i >= 0:
		wordDist[i] = (words[i], i-count)
		i -= 1
	while j < len(words):
		wordDist[j] = (words[j], j-count)
		j += 1
	return wordDist

# I don't know the behavior when I observe more than one predicate in a sentence
def constructPredicateDistanceVectors(wordAttr):
	verbs =	{'VB': 1, 'VBD': 1, 'VBG': 1, 'VBN': 1,	'VBP': 1, 'VBZ': 1}
	predWords = [attr[0] for attr in wordAttr]
	count = 0
	ans = []
	for attr in wordAttr:
		if attr[2] in verbs:
			ans.append(getDistanceTuple(predWords, count))
		count += 1
	return ans

def constructImportantWordDistanceVectors(wordAttr, wordOfImportance):
	argWords = [attr[0] for attr in wordAttr]
	count = 0
	for attr in wordAttr:
		if attr[0] == wordOfImportance:
			 return getDistanceTuple(argWords, count)
		count += 1
	return None

if  __name__ == '__main__':
	wordAttr = getWordAttributes('../resources/sentence.txt')
	print constructPredicateDistanceVectors(wordAttr)
	print constructImportantWordDistanceVectors(wordAttr, 'robot')

