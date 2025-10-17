BLK         =   black
MAKE        =   make
PIP         =   pip3

BASEDIR     =   $(CURDIR)
SRCDIR      =   ${BASEDIR}/fitpdf

help:
	@echo 'Makefile for fitpdf'
	@echo 'Usage:'
	@echo 'make black           reformat the code using black code formatter'
	@echo 'make clean           remove temporary files'
	@echo 'make install         install the package locally'
	@echo 'make uninstall       uninstall the package'
	@echo 'make upload          upload the distribution to PyPI'
	@echo 'make uploadtest      upload the distribution to TestPyPI'

black:
	${BLK} *.py */*.py */*/*.py

build:
	python3 -m build

clean:
	rm -f ${SRCDIR}/*.pyc
	rm -f ${SRCDIR}/apps/*.pyc
	rm -rf ${SRCDIR}/__pycache__
	rm -rf ${SRCDIR}/apps/__pycache__
	rm -rf ${BASEDIR}/build
	rm -rf ${BASEDIR}/dist
	rm -rf ${BASEDIR}/fitpdf.egg-info

install:
	${MAKE} clean
	${MAKE} uninstall
	${PIP} install .
	${MAKE} clean

uninstall:
	${PIP} uninstall --yes fitpdf

upload:
	${MAKE} clean
	${MAKE} build
	python3 -m twine upload dist/*

uploadtest:
	${MAKE} clean
	${MAKE} build
	python3 -m twine upload --repository testpypi dist/*

.PHONY: help black build clean install uninstall upload uploadtest