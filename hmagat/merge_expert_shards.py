import argparse

from loguru import logger

from hmagat.expert_shards import merge_expert_shards
from hmagat.run_expert import add_expert_dataset_args


def main():
    parser = argparse.ArgumentParser(description="Merge sharded expert datasets")
    parser = add_expert_dataset_args(parser)
    parser.add_argument("--num_shards", type=int, default=None)

    args = parser.parse_args()
    logger.info(args)
    merge_expert_shards(args, expected_num_shards=args.num_shards)


if __name__ == "__main__":
    main()
