

datasets = ['train', 'eval']
dataclasses = ['joy', 'fear', 'sadness', 'anger']

for dataset in datasets:
	for dataclass in dataclasses:
		with open('./{}/{}-ori.txt'.format(dataset, dataclass)) as f:
			sentences = f.readlines()
		for i, sentence in enumerate(sentences):
			if i == 0:
				continue
			data = sentence.split('\t')
			#print i, sentence, data
			assert(data[-2] == dataclass)			
			sentences[i] = ' '.join(data[1:-2])

		f = open('./{}/{}.txt'.format(dataset, dataclass), 'w')
		for i, sentence in enumerate(sentences):
			if i == 0:
				continue
			f.write(sentence + '\n')
		f.close()