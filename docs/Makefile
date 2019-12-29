# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS            =
SPHINXBUILD           = python -msphinx
SPHINXPROJ            = minkit
SOURCEDIR             = source
BUILDDIR             := $(if $(BUILDDIR),$(BUILDDIR),../../minkit-docs)
INPUT_NOTEBOOKS_DIR   = notebooks
OUTPUT_NOTEBOOKS_DIR  = source/$(INPUT_NOTEBOOKS_DIR)
GITCOMMANDS           = cd $(BUILDDIR)/html; git add . ; git commit -m 'Rebuilt docs' ; git push origin gh-pages
PROJECT_SOURCES       = $(shell find ../minkit -type f)

ALLNOTEBOOKS      = $(wildcard $(INPUT_NOTEBOOKS_DIR)/*.ipynb)
ALLNOTEBOOKS_OUT = $(patsubst $(INPUT_NOTEBOOKS_DIR)/%.ipynb, $(OUTPUT_NOTEBOOKS_DIR)/%.ipynb, $(ALLNOTEBOOKS))

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo 'If the target is "gh-pages", "html" will be built, and the '\
	'result will be automatically published through the command'
	@echo "$(GITCOMMANDS)"

.PHONY: help Makefile

# Target to build the GitHub Pages
gh-pages: html
	$(GITCOMMANDS)

# Create the directory to contain the notebooks
$(OUTPUT_NOTEBOOKS_DIR):
	mkdir -p $@

# Target to update the notebooks
$(OUTPUT_NOTEBOOKS_DIR)/%.ipynb: $(INPUT_NOTEBOOKS_DIR)/%.ipynb $(OUTPUT_NOTEBOOKS_DIR) $(PROJECT_SOURCES)
	jupyter nbconvert --execute --to notebook --output $(subst $(OUTPUT_NOTEBOOKS_DIR)/,,$@) --output-dir $(OUTPUT_NOTEBOOKS_DIR) $<

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: Makefile $(ALLNOTEBOOKS_OUT) $(PROJECT_SOURCES)
	@if [ -d $(BUILDDIR)/doctrees ]; then rm -r $(BUILDDIR)/doctrees; fi
	@if [ -d source/api ]; then rm -r source/api; fi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
