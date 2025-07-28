SHELL     := bash
MAKEFLAGS += --warn-undefined-variables
.SILENT:

LOUD = \033[1;34m#
HIGH = \033[1;33m#
SOFT = \033[0m#

Top=$(shell git rev-parse --show-toplevel)

help:  ## show help
	gawk -f $(Top)/etc/help.awk $(MAKEFILE_LIST) 

pull: ## update from main
	git pull

push: ## commit to main
	- echo -en "$(LOUD)Why this push? $(SOFT)" 
	- read x ; git commit -am "$$x" ;  git push
	- git status

sh: ## run my shell
	bash --init-file  $(Top)/etc/dotbash.rc -i

lint: ## lint all python in this directory
	export PYTHONPATH="..:$$PYTHONPATH"; \
	pylint --disable=W0311,C0303,C0116,C0321,C0103 \
		    --disable=C0410,C0115,C3001,R0903,E1101 \
		    --disable=E701,W0108,W0106,W0718,W0201  *.py

~/tmp/%.pdf: %.py ## pdf print Python
	echo "making $@"
	a2ps                     \
  --file-align=virtual      \
	--line-numbers=1           \
	--pro=color                 \
	--pretty=python              \
	--chars-per-line=80           \
	--left-title=""                \
	--borders=no                    \
  --right-footer='page %s. of %s#' \
	--portrait                        \
	--columns 2                        \
	-M letter                           \
	-o - $^ | ps2pdf - $@
	open $@
