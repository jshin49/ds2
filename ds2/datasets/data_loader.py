import json
import random

from collections import defaultdict
from typing import Dict
from typing import List

from torch.utils.data import DataLoader, Dataset

from ds2.utils.few_shot import get_balanced_fewshot_samples
from ds2.utils.few_shot import get_filtered_fewshot_samples
from ds2.utils.few_shot import get_final_states
from ds2.utils.fix_label import fix_general_label_error
from ds2.utils.fix_label import has_or_character
from ds2.utils.state_sum_converter import get_converter

EXPERIMENT_DOMAINS = set(["hotel", "train", "restaurant", "attraction", "taxi"])
EXCLUDE_DOMAINS = set(["hospital", "police"])


class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]

        return item_info

    def __len__(self):
        return len(self.data)


def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    # Generate domain-dependent slot list
    if args["only_domain"]:
        eval_slots = [k for k in SLOTS if k.startswith(args["only_domain"])]
    elif args['except_domain']:
        eval_slots = [k for k in SLOTS if not k.startswith(args["except_domain"])]
    else:
        eval_slots = SLOTS

    print(("Reading all files from {}".format(path_name)))
    data = []

    converter = get_converter(args['state_converter'])

    domain_counter = defaultdict(int)
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        for dial_dict in dials:
            dialog_history = ""

            # Skip if the domain is in (hospital, police) as they are not in test set
            # (hospital, police) also only occur as single domain
            if dial_dict["domains"][0] in EXCLUDE_DOMAINS:
                continue
            
            # Counting domains
            for domain in dial_dict["domains"]:
                if domain in EXPERIMENT_DOMAINS:
                    domain_counter[domain] += 1

            # Dialogue-level filtering
            if args["only_domain"]:
                if args["only_domain"] not in dial_dict["domains"]:
                    continue
            elif args["except_domain"]:
                """
                There are two options to filter dialogue samples when pre-training a model with a given except_domain.
                max: Filter out every dialogue that contains the except_domain context.
                min: Filter out a dialogue only if except_domain is the one and only domain that it has.
                """
                if args["dialogue_filter"] == "max" and args["except_domain"] in dial_dict["domains"]:
                    continue
                elif args["dialogue_filter"] == "min" and [args["except_domain"]] == dial_dict["domains"]:
                    continue

            # Reading data
            for turn_id, turn in enumerate(dial_dict["turns"]):
                # accumulate dialogue utterances
                dialog_history += (
                    " system: " + turn["system"] + " user: " + turn["user"]
                )

                slot_values = fix_general_label_error(turn["state"]['slot_values'], SLOTS)
                slot_values = {k: v for k, v in slot_values.items() if v != "none"}

                if dataset in {"train", "dev"} and has_or_character(slot_values):
                    continue

                if args["except_domain"] and any([k.startswith(args["except_domain"]) for k in slot_values]):
                    continue


                if args["model_name"] == "t5":
                    # Our t5-large-samsum model is trained T0 style which has the following source prefix
                    input_text = f"Summarize this dialogue: {dialog_history.lower()} {tokenizer.eos_token}"
                else:
                    input_text = f"{tokenizer.bos_token} {dialog_history.lower()} {tokenizer.eos_token}"

                eval_slots_per_sample = set(s for s in eval_slots if s.split("-")[0] in dial_dict["domains"])

                data_detail = {
                    "ID": dial_dict["dial_id"],
                    "domains": dial_dict["domains"],
                    "turn_id": turn_id,
                    "dialog_history": dialog_history,
                    "intput_text": input_text,
                    "slot_values": slot_values,
                    "eval_slots": eval_slots_per_sample,
                }
                if dataset in {"dev", "test"}:
                    output_text = converter.state_to_sum(slot_values)
                    data_detail["output_text"] = output_text
                data.append(data_detail)

    print("domain_counter", domain_counter)
    return data


def get_slot_information(ontology: Dict[str, List[str]]) -> List:
    ontology_domains = dict(
        [(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS]
    )
    SLOTS = [
        k.replace(" ", "").lower() if ("book" not in k) else k.lower()
        for k in ontology_domains.keys()
    ]

    return SLOTS


def collate_fn(tokenizer, converter):
    def _collate(batch):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [d[key] for d in batch]

        input_batch = tokenizer(
            batch_data["intput_text"],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
            truncation=True,
            max_length=1000,
        )

        if "output_text" not in batch_data:
            batch_data["output_text"] = [converter.state_to_sum(x) for x in batch_data["slot_values"]]

        batch_data["encoder_input"] = input_batch["input_ids"]
        batch_data["attention_mask"] = input_batch["attention_mask"]
        batch_data["decoder_output"] = tokenizer(
            batch_data["output_text"],
            padding=True,
            return_tensors="pt", # non-padded return List[List[Int]]
            return_attention_mask=False,
            truncation=True,
            max_length=200,
        ).input_ids

        return batch_data
    return _collate


def normalize_ontology(ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
    keys = [k for k in ontology]
    for k in keys:
        for i in range(len(ontology[k])):
            ontology[k][i] = ontology[k][i].replace("do n't care", "dontcare")
            ontology[k][i] = ontology[k][i].replace("'s", " s")

        ontology[
            k.replace(" ", "").lower() if ("book" not in k) else k.lower()
        ] = ontology.pop(k)

    return ontology


def prepare_data(args, tokenizer):
    if args["version"] == "2.0":
        paths = {
            k: f"ds2/data_mwoz_2.0/{k}_dials.json"
            for k in ("train", "dev", "test")
        }
        ontology = normalize_ontology(json.load(open("ds2/data_mwoz_2.0/mwz/ontology.json", "r")))
    else:
        paths = {
            k: f"ds2/data_mwoz_2.1/{k}_dials.json"
            for k in ("train", "dev", "test")
        }
        ontology = normalize_ontology(json.load(open("ds2/data_mwoz_2.1/mwz/ontology.json", "r")))

    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open("ds2/utils/slot_description.json", "r"))

    datasets = {
        k: DSTDataset(read_data(args, path, ALL_SLOTS, tokenizer, description, k), args)
        for k, path in paths.items()
    }
    
    if 0.0 < args["fewshot"] < 1.0:
        num_train_diags = len(set(x["ID"] for x in datasets["train"]))
        num_few_diags = int(num_train_diags * args["fewshot"])

        final_states = get_final_states(datasets["train"])

        if args["filtered_sampling"]:
            final_states = get_filtered_fewshot_samples(final_states)

        if args["balanced_sampling"]:
            sampled_ids = get_balanced_fewshot_samples(final_states, num_few_diags, ALL_SLOTS)
        else:
            sampled_ids = random.sample(final_states.keys(), num_few_diags)

        datasets["train"].data = [x for x in datasets["train"].data if x["ID"] in sampled_ids]

        domain_counter = defaultdict(int)
        multi_domain_counter = defaultdict(int)
        for d in datasets["train"].data:
            domains = d["domains"]
            multi_domain_counter[tuple(sorted(d["domains"]))] += 1
            for domain in domains:
                if domain in EXPERIMENT_DOMAINS:
                    domain_counter[domain] += 1
        print("num_train_diags", num_train_diags, len(sampled_ids))
        print("domain_counter", domain_counter)
        print("multi_domain_counter", multi_domain_counter)

    print(f'dontcare occurence: {sum(["dontcare" in x["slot_values"].values() for x in datasets["train"].data]) / len(datasets["train"].data)}')

    if args["debug_code"]:
        datasets["train"] = datasets["train"][:50]
        datasets["dev"] = datasets["dev"][:50]
        datasets["test"] = datasets["test"][:50]

    dataloaders = {
        k: DataLoader(
            dataset,
            batch_size=args[f"{k}_batch_size"],
            shuffle=(k == "train"),
            collate_fn=collate_fn(tokenizer=tokenizer, converter=get_converter(args["state_converter"])),
        ) for k, dataset in datasets.items()
    }

    domain_data = {}
    return dataloaders, domain_data

