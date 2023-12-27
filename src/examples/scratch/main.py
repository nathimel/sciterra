from examples.scratch import util

from sciterra.mapping.tracing import AtlasTracer


def main(args):
    seed = args.seed
    util.set_seed(seed)

    tracer = AtlasTracer(
        atlas_dir=args.atlas_dir,
        atlas_center_bibtex=args.bibtex_fp,
        librarian_name=args.api,
        vectorizer_name=args.vectorizer,
        vectorizer_kwargs={
            "device": "mps",
            "model_path": args.model_path,
        },
    )

    tracer.expand_atlas(
        target_size=args.target_size,
        max_failed_expansions=args.max_failed_expansions,
        n_pubs_max=args.max_pubs_per_expand,
        call_size=args.call_size,
        record_pubs_per_update=True,
    )


if __name__ == "__main__":
    args = util.get_args()

    main(args)
