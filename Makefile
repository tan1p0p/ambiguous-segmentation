build:
	docker build . -t ambiguous-segmentation
run:
	docker run -v `pwd`:/workspace -it ambiguous-segmentation
train:
	python scripts/main.py
test:
	pytest scripts/test.py -s --pdb
