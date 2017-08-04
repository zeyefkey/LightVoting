import argparse
from types import MappingProxyType
import yaml
import numpy
from numpy.random import RandomState


PRNG = RandomState()


def uniform_distribution(config):
    return {
        "type": "uniform",
        "category_preferences": numpy.round(
            PRNG.rand(len(config.get("categories"))), config.get("precission")
        )
    }


def cluster_distribution(atype, config):
    return {
        "category_preferences": numpy.round(
            numpy.clip(
                PRNG.normal(
                    config.get("distributions").get("mu"),
                    config.get("distributions").get("sigma"),
                    len(config.get("categories").keys())
                ) * numpy.array(
                    [
                        v for k, v in sorted(
                            config.get("distributions").get("preferences").get(atype).items(),
                            key=lambda v: v[0])
                    ]
                ),
                0,
                1
            ),
            config.get("precission")
        ).tolist(),
        "type": str(atype)
    }


def poi_preferences(categories, category_preferences, config):
    return {
        **category_preferences,
        "poi_preferences": numpy.round([
            i_pref * config.get("categories").get(i_cat).get(i_poi).get("popularity")/10
            for i_cat, i_pref in zip(categories, category_preferences.get("category_preferences"))
            for i_poi in config.get("categories").get(i_cat)
        ], config.get("precission")).tolist()
    }


def read_config(configfile):
    with open(configfile, "r") as f_config:
        return MappingProxyType(yaml.load(f_config, Loader=yaml.CSafeLoader))


def write_runconfig(config, args):
    with open(args.outputfile, "w") as f_output:
        categories = sorted([i_category for i_category in config.get("categories").keys()])
        pois = sorted(
            i_poi for i_cat in config.get("categories")
            for i_poi in config.get("categories").get(i_cat)
        )
        yaml.dump(
            {
                "categories": categories,
                "pois": pois,
                "run": {
                    "number_"+str(i_run): {
                        "agent_" + str(i_agent):
                            poi_preferences(
                                categories=categories,
                                category_preferences=cluster_distribution(
                                    PRNG.choice(
                                        list(config.get("distributions").get("preferences").keys()),
                                        p=config.get("agentdist")
                                    ),
                                    config
                                ) if args.cluster else uniform_distribution(config),
                                config=config
                                )
                        for i_agent in range(config.get("agents"))
                    }
                    for i_run in range(config.get("runs"))
                }
            },
            f_output,
            Dumper=yaml.CSafeDumper
        )

def main():
    parser = argparse.ArgumentParser(prog="lightvoting_prefgen.py")
    parser.add_argument("--config", dest="configfile", type=str, default="configuration.yaml")
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument("--uniform", dest="uniform", action="store_true", default=False)
    mutex_group.add_argument("--cluster", dest="cluster", action="store_true", default=True)
    parser.add_argument("--output", dest="outputfile", type=str, default="runconfig.yaml")
    args = parser.parse_args()
    config = read_config(args.configfile)

    write_runconfig(config, args)

if __name__ == "__main__":
    main()
