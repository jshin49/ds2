import json
import random
from collections import defaultdict

from transformers import T5Tokenizer

from ds2.configs.config import get_args

EXCEPT_VALUES = {
    "hotel-parking": ["free"],
    "hotel-book stay": ["six"],
    "hotel-book people": ["six", "3."],
    "train-destination": ["no"],
    "train-day": [
        "fr",
        "n",
        "we",
        "train"
    ],
    "train-departure": [
        "a",
    ],
    "train-arrive by": [
        "thursday",
        "1100",
        "tuesday",
    ],
    "train-book people": [
        "`1",
    ],
    "train-leave at": [
        "after 9:15 am",
        "bishops stortford",
        "after 11:30",
        "do n't care",
        "0",
        "monday",
        "leicester",
        "after 8:30",
        "after 19:30",
        "morning",
        "saturday",
        "after 16:30",
        "thursday",
    ],
    "restaurant-food": [
        "kor",
        "steakhouse",
    ],
    "restaurant-area": [
        "centre",
        "south",
        "north",
        "east",
        "west"
    ],
    "attraction-name": [
        "no",
        "ay",
        "fi",
        "free",
        "bo",
        "by",
        "s",
        "c",
        "ac",
        "k",
        "mi"
    ],
    "restaurant-name": [
        "no",
        "golden house                            golden house",
        "au",
        "el",
        "yes",
        "d",
        "hu",
    ],
    "hotel-name": [
        "to",
        "n",
        "a",
        "hotel",
        "yes",
        "me",
        "nus",
        "gra",
        "the",
        "ca"
    ],
    "taxi-leave at": [
        "after 2:30",
        "1615",
        "after 11:45",
        "thursday",
        "after 15:45",
        "300",
        "monday",
        "friday",
        "1"
    ],
    "taxi-destination": [
        "shiraz.",
    ],
    "taxi-departure": [
        "w",
        "g",
        "ca",
        "07:15",
    ],
    "restaurant-book time": [
        "1545",
        "1745",
        "not given",
    ],
    "restaurant-book day": [
        "thur",
        "not given",
        "w"
    ],
    "taxi-arrive by": [
        "1145",
        "1730",
        "1700",
    ],
}


def get_balanced_fewshot_samples(final_states: dict, num_target_samples: int, ALL_SLOTS) -> dict:
    """
    final_states: a dictioinary whose keys are dialogue ID's and values are states of the final turn
    """
    sampled_slots_count = {_slot_name: 0 for _slot_name in ALL_SLOTS}
    sampled_ids = []

    domains = {
        dial_id: tuple(sorted(set(_slot.split("-")[0] for _slot in ds))) for dial_id, ds in final_states.items()
    }
    sampled_domains_count = {_domain: 0 for _domain in set(domains.values())}
    sampled = random.choice([_dial_id for _dial_id, _domain in domains.items() if len(_domain) == 1])
    sampled_ids.append(sampled)
    del domains[sampled], final_states[sampled]

    while len(sampled_ids) < num_target_samples:
        target_slot = get_argmin(sampled_slots_count)
        candidates = [_dial_id for _dial_id, _state in final_states.items() if target_slot in _state]

        if len(candidates) == 0:
            del sampled_slots_count[target_slot]
            continue

        sampled = random.choice(candidates)
        sampled_ids.append(sampled)

        for _slot in final_states[sampled]:
            if _slot in sampled_slots_count:
                sampled_slots_count[_slot] += 1
        sampled_domains_count[domains[sampled]] += 1

        del final_states[sampled], domains[sampled]

        if len(sampled_ids) >= num_target_samples:
            break

        target_domain = get_argmin(sampled_domains_count)
        candidates = [_dial_id for _dial_id, _domain in domains.items() if _domain == target_domain]
        if len(candidates) == 0:
            del sampled_domains_count[target_domain]
            continue
        sampled = random.choice(candidates)
        sampled_ids.append(sampled)

        for _slot in final_states[sampled]:
            if _slot in sampled_slots_count:
                sampled_slots_count[_slot] += 1
        sampled_domains_count[domains[sampled]] += 1

        del domains[sampled], final_states[sampled]

    print(sampled_slots_count.values())
    print(sampled_domains_count.values())
    return sampled_ids


def get_filtered_fewshot_samples(final_states: dict) -> dict:
    """
    final_states: a dictioinary whose keys are dialogue ID's and values are states of the final turn
    """
    return {dial_id: ds for dial_id, ds in final_states.items() if not has_except_value(ds)}


def get_final_states(dataset) -> dict:
    final_turn_ids = defaultdict(int)
    final_turn_states = {}
    for x in dataset:
        dial_id, turn_id = x["ID"], x["turn_id"]
        if final_turn_ids[dial_id] < turn_id:
            final_turn_ids[dial_id] = turn_id
            final_turn_states[dial_id] = x["slot_values"]

    return final_turn_states


def get_argmin(d):
    min_v = min(d.values())
    for k, v in d.items():
        if v == min_v:
            return k


def has_except_value(ds):
    return any(
        ds[_key] in EXCEPT_VALUES[_key]
        for _key in set(ds.keys()) & set(EXCEPT_VALUES.keys())
    )


# if __name__ == "__main__":
#     from datasets.data_loader import DSTDataset
#     from datasets.data_loader import get_slot_information
#     from datasets.data_loader import normalize_ontology
#     from datasets.data_loader import read_data
#
#     path = "data/train_dials.json"
#     ontology = normalize_ontology(json.load(open("datasets/synth_ontology.json", "r")))
#     ALL_SLOTS = get_slot_information(ontology)
#     description = json.load(open("utils/slot_description.json", "r"))
#     args = vars(get_args())
#     tokenizer = T5Tokenizer.from_pretrained(
#         "t5-small", bos_token="[bos]", eos_token="[eos]", sep_token="[sep]"
#     )
#
#     dataset = DSTDataset(read_data(args, path, ALL_SLOTS, tokenizer, description), args)
#
#     final_states = get_final_states(dataset)
#
#     final_states = get_filtered_fewshot_samples(final_states)
#
#     sampled_ids = get_balanced_fewshot_samples(final_states, 20, ALL_SLOTS)
#
#     dataset.data = [x for x in dataset if x["ID"] in sampled_ids]
