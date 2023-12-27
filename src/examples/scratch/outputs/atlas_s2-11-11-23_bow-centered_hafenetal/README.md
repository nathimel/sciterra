# README

This was the first atlas generated with a bag-of-words embedding method.

The command was:

python main.py --bibtex_fp data/hafenLowredshiftLymanLimit2017.bib --atlas_dir outputs/atlas_s2-11-11-23_bow-centered_hafenetal --model_path outputs/atlas_s2-11-11-23_w2v-centered_hafenetal/astro_1.model --centered True --max_pubs_per_expand 500 --vectorizer BOW --api S2 --call_size 10 --target_size 10000
