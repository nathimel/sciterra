# README

This was the first atlas that 'fully' replicated Zach's experiments, in the sense that we expanded iteratively until 10k pubs, and we successfully tracked the convergence degree and filtered publications based on this later. While locally the convergence trendlines look like Zach's, they don't stabilize as well necessary, in the sense that they jump after stabilizing. I suspect this is because while Zach consistently updated roughly 4000 pubs at a time, we are only updating by about 150 pubs each iteration, due to missing data and what seems to be rate-limits.

The command was

python main.py --bibtex_fp data/hafenLowredshiftLymanLimit2017.bib --atlas_dir outputs/atlas_s2-11-10-23_centered_hafenetal --centered True --max_pubs_per_expand 500 --api S2 --call_size 10 --target_size 10000

I have since added a new vectorization method, so we'll probably let SciBERT be the default vectorizer, but add an cml arg.
