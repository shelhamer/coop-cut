.PHONY: clean

pymaxflow: maxflow
	python setup.py build_ext -i 

maxflow: 
	wget http://pub.ist.ac.at/~vnk/software/maxflow-v3.03.src.zip
	mkdir maxflow
	cd maxflow && unzip ../maxflow-v3.03.src.zip

clean:
	@ $(RM) -f pymaxflow.cpp pymaxflow.so
