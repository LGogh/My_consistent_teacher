pre:
	python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	mkdir -p thirdparty
	git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection
	cd thirdparty/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info


# git checkout ca11860f4f3c3ca2ce8340e2686eeaec05b29111  mmdet==2.25.0	
