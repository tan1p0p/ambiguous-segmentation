build:
	docker build . -t ambiguous-segmentation
run:
	docker run -v /home/s1511526/ambiguous-segmentation:/workspace -it ambiguous-segmentation