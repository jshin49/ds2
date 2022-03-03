import random
import re

from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Set



def safeint(x) ->int:
    try:
        return int(x)
    except:
        return 0

DOMAIN_PHRASE_IN_SENTENCE = {
    "train": "a train",
    "taxi": "a taxi",
    "restaurant": "a restaurant",
    "hotel": "a place to stay",
    "attraction": "an attraction"
}

DOMAIN_SLOT_TEMPLATES = {
    "train": OrderedDict([
        ("train-book people", lambda x, either: f"for {either(x)} {'people' if safeint(x) > 1 else 'person'}"),
        ("train-departure", lambda x, either: f"from {either(x)}"),
        ("train-destination", lambda x, either: f"to {either(x)}"),
        ("train-day", lambda x, either: f"on {either(x)}"),
        ("train-leaveat", lambda x, either: f"leaves at {either(x)}"),
        ("train-arriveby", lambda x, either: f"arrives by {either(x)}"),
    ]),
    "taxi": OrderedDict([
        ("taxi-departure", lambda x, either: f"from {either(x)}"),
        ("taxi-destination", lambda x, either: f"to {either(x)}"),
        ("taxi-leaveat", lambda x, either: f"leaves at {either(x)}"),
        ("taxi-arriveby", lambda x, either: f"arrives by {either(x)}"),
    ]),
    "restaurant": OrderedDict([
        ("restaurant-name", lambda x, either: f"called {either(x)}"),
        ("restaurant-area", lambda x, either: f"located in the {either(x)}"),
        ("restaurant-food", lambda x, either: f"serves {either(x)}"),
        ("restaurant-pricerange", lambda x, either: f"with {either(f'an {x}' if any([x.startswith(prefix) for prefix in ['a', 'e', 'o', 'u', 'i']]) else f'a {x}')} price"),
        ("restaurant-book people", lambda x, either: f"for {either(x)} {'people' if safeint(x) > 1 else 'person'}"),
        ("restaurant-book day", lambda x, either: f"on {either(x)}"),
        ("restaurant-book time", lambda x, either: f"at {either(x)}"),
    ]),
    "attraction": OrderedDict([
        ("attraction-type", lambda x, either: f"which is {either(f'an {x}' if any([x.startswith(prefix) for prefix in ['a', 'e', 'o', 'u', 'i']]) else f'a {x}')}"),
        ("attraction-name", lambda x, either: f"called {either(x)}"),
        ("attraction-area", lambda x, either: f"located in the {either(x)}"),
    ]),
    "hotel": OrderedDict([
        ("hotel-type", lambda x, either: f"which is a {either(x)}"),
        ("hotel-name", lambda x, either: f"called {either(x)}"),
        ("hotel-stars", lambda x, either: f"ranked {either(x)} stars"),
        ("hotel-pricerange", lambda x, either: f"with {either(f'an {x}' if any([x.startswith(prefix) for prefix in ['a', 'e', 'o', 'u', 'i']]) else f'a {x}')} price"),
        ("hotel-area", lambda x, either: f"located in the {either(x)}"),
        ("hotel-book people", lambda x, either: f"for {either(x)} {'people' if safeint(x) > 1 else 'person'}"),
        ("hotel-book day", lambda x, either: f"on {either(x)}"),
        ("hotel-book stay", lambda x, either: f"for {either(x)} day{'s' if safeint(x) > 1 else ''}"),
        ("hotel-parking", lambda x, either: f"has {either('no ' if x == 'no' else '')}parking"),
        ("hotel-internet", lambda x, either: f"has {either('no ' if x == 'no' else '')}internet"),
    ])
}

DOMAIN_DONTCARE_PHRASES_DICT = {
    "train": OrderedDict([
        ("train-book people", "the number of people"),
        ("train-departure", "the point of departure"),
        ("train-destination", "the destination"),
        ("train-day", "the departure date"),
        ("train-arriveby", "the arrival time"),
        ("train-leaveat", "the departure time"),
    ]),
    "restaurant": OrderedDict([
        ("restaurant-name", "the name"),
        ("restaurant-area", "the location"),
        ("restaurant-food", "the food type"),
        ("restaurant-pricerange", "the price range"),
        ("restaurant-book people", "the number of people"),
        ("restaurant-book day", "the day"),
        ("restaurant-book time", "the time"),
    ]),
    "hotel": OrderedDict([
        ("hotel-type", "the hotel type"),
        ("hotel-name", "the name"),
        ("hotel-stars", "the hotel stars"),
        ("hotel-pricerange", "the price range"),
        ("hotel-area", "the location"),
        ("hotel-book people", "the number of people"),
        ("hotel-book day", "the day"),
        ("hotel-book stay", "the stay"),
        ("hotel-parking", "the parking"),
        ("hotel-internet", "the internet"),
    ]),
    "taxi": OrderedDict([
        ("taxi-destination", "the destination"),
        ("taxi-departure", "the point of departure"),
        ("taxi-leaveat", "the departure time"),
        ("taxi-arriveby", "the arrival time"),
    ]),
    "attraction": OrderedDict([
        ("attraction-area", "the location"),
        ("attraction-name", "the name of the attraction"),
        ("attraction-type", "the type of the attraction"),
    ]),
}

COMMON_PHRASES = [
    "is looking for",
    "is searching for",
    "looks for",
    "searches for",
    "wants",
]


def get_first_sentence(ds: Dict, domain: str, either: callable, except_keys: Set[str], idx: int, wo_para: bool):
    # For example, if ds = {"hotel-type": "hotel", "hotel-stars": "3", "hotel-book people": "1", ... },
    # slot_phrases = {"hotel-type": "which is a hotel", "hotel-stars": "ranked 3 stars", "hotel-book people": "for 1 person", ... }
    slot_phrases = {
        _slot_name: f(ds[_slot_name], either) for _slot_name, f in DOMAIN_SLOT_TEMPLATES[domain].items()
        if _slot_name in ds and ds[_slot_name] != "dontcare"
    }

    # example: f"{'The user'} {'is looking for'} {'a place to stay'} {'which is a hotel'} ... "
    # example: f"{'he'} {'is searching for'} {'a place to stay'} {'which is a guesthouse'} {'called Ocean house'} ... "
    sentence_idx = 0 if wo_para else idx
    res = " ".join(
        [f"{'The user' if sentence_idx == 0 else 'he'} {COMMON_PHRASES[sentence_idx]} {DOMAIN_PHRASE_IN_SENTENCE[domain]}"] + [
            _phrase for _slot_name, _phrase in slot_phrases.items()
            if _slot_name not in except_keys
        ]
    )

    # example: ["has internet", "has no parking"]
    rest_phrases = [
        _phrase for _slot_name, _phrase in slot_phrases.items()
        if _slot_name in except_keys
    ]
    if rest_phrases:
        # example: ", which has internet and has no parking"
        res += f", which "
        res += " and ".join(rest_phrases)

    return res


def get_dontcare_sentence(ds: dict, domain: str, either: callable, is_one_sentence: bool, wo_para: bool):
    dontcare_phrases = [
        either(_phrase) for _slot_name, _phrase in DOMAIN_DONTCARE_PHRASES_DICT[domain].items()
        if _slot_name in ds and ds[_slot_name] == "dontcare"
    ]

    if len(dontcare_phrases) > 0:
        if wo_para:
            second_sentence = f'{", and the user" if is_one_sentence else ". The user"} does not care about '
        else:
            second_sentence = f'{", and he" if is_one_sentence else ". He"} does not care about '

        if len(dontcare_phrases) == 1:
            second_sentence += dontcare_phrases[0]
        elif len(dontcare_phrases) == 2:
            second_sentence += f"{dontcare_phrases[0]} and {dontcare_phrases[1]}"
        else:
            dontcare_phrases[-1] = f"and {dontcare_phrases[-1]}"
            second_sentence += ', '.join(dontcare_phrases)
    else:
        second_sentence = ""

    return second_sentence


def get_dontcare_values(summary: str, domain: str):
    dontcare_match = re.search("does not care about", summary)
    if dontcare_match:
        rest = summary[dontcare_match.span()[1]:]
        return {
            _slot_name: "dontcare"
            for _slot_name, _phrase in DOMAIN_DONTCARE_PHRASES_DICT[domain].items()
            if re.search(_phrase, rest)
        }
    else:
        return {}


def train_state_to_sum(ds: dict, either: callable, is_one_sentence: bool, idx: int, wo_para: bool) -> str:
    """
    Input:
        example: {'train-departure': 'london', 'train-destination': 'cambridge', 'train-arriveby': '12:30', 'train-book people': '3', 'train-day': 'tuesday'}
    Returns:
        example: "The user is looking for a train for 3 people from london to cambridge on tuesday, which arrives by 12:30."
    """
    first_sentence = get_first_sentence(ds, domain="train", either=either, except_keys={"train-arriveby", "train-leaveat"}, idx=idx, wo_para=wo_para)

    second_sentence = get_dontcare_sentence(
        ds,
        domain="train",
        either=either,
        is_one_sentence=is_one_sentence,
        wo_para=wo_para
    )
    res = first_sentence + second_sentence + "."

    return res


def taxi_state_to_sum(ds: Dict, either: callable, is_one_sentence: bool, idx: int, wo_para: bool) -> str:
    """
    Input:
        example: {"taxi-destination": 'galleria', "taxi-departure": 'kirkwood house', "taxi-leaveat": '12:30', "taxi-arriveby": "19:15"}
    Returns:
        example: "The user is looking for a taxi from kirkwood house to galleria, which arrives by 19:15 and leaves at 12:30."
    """
    first_sentence = get_first_sentence(ds, domain="taxi", either=either, except_keys={"taxi-arriveby", "taxi-leaveat"}, idx=idx, wo_para=wo_para)

    second_sentence = get_dontcare_sentence(
        ds,
        domain="taxi",
        either=either,
        is_one_sentence=is_one_sentence,
        wo_para=wo_para
    )
    res = first_sentence + second_sentence + "."

    return res


def restaurant_state_to_sum(ds: Dict, either: callable, is_one_sentence: bool, idx: int, wo_para: bool) -> str:
    """
    Input:
        example: {
                'restaurant-book day': 'tuesday', 'restaurant-book people': '6',
                'restaurant-book time': '12:00', 'restaurant-name': 'meze bar', 'restaurant-pricerange': 'cheap',
                'restaurant-area': 'south', 'restaurant-food': 'seafood',
            }
    Returns:
        example: "The user is looking for a restaurant called meze bar located in south,
            which serves seafood under cheap price range
            for 6 people on tuesday at 12:00.
            "
    """
    first_sentence = get_first_sentence(ds=ds, domain="restaurant", either=either, except_keys={"restaurant-food"}, idx=idx, wo_para=wo_para)

    second_sentence = get_dontcare_sentence(
        ds,
        domain="restaurant",
        either=either,
        is_one_sentence=is_one_sentence,
        wo_para=wo_para
    )
    res = first_sentence + second_sentence + "."

    return res


def attraction_state_to_sum(ds: Dict, either: callable, is_one_sentence: bool, idx: int, wo_para: bool) -> str:
    """
    Input:
        example: {
            "attraction-area": "cambridge",
            "attraction-name": "nusha",
            "attraction-type": "museum",
            }
    Returns:
        example: "The user is looking for an attraction called nusha, which is a museum located in cambridge."
    """
    first_sentence = get_first_sentence(ds, domain="attraction", either=either, except_keys=set(), idx=idx, wo_para=wo_para)

    second_sentence = get_dontcare_sentence(
        ds,
        domain="attraction",
        either=either,
        is_one_sentence=is_one_sentence,
        wo_para=wo_para
    )
    res = first_sentence + second_sentence + "."

    return res


def hotel_state_to_sum(ds: Dict, either: callable, is_one_sentence: bool, idx: int, wo_para: bool) -> str:
    first_sentence = get_first_sentence(ds=ds, domain="hotel", either=either, except_keys={"hotel-parking", "hotel-internet"}, idx=idx, wo_para=wo_para)

    second_sentence = get_dontcare_sentence(
        ds,
        domain="hotel",
        either=either,
        is_one_sentence=is_one_sentence,
        wo_para=wo_para
    )

    res = first_sentence + second_sentence + "."
    return res


def train_sum_to_state(summ: str , is_one_sentence: bool) -> dict:
    sentences = re.split("|".join(COMMON_PHRASES), summ)
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["train"] in sentence]
    if not summary:
        return {}
    summary = summary[0]
    slot_to_prefix = {
        "train-departure": " from ",
        "train-destination": " to ",
        "train-arriveby": " arrives by ",
        "train-leaveat": " leaves at ",
        "train-book people": r" for \d+ p",
        "train-day": " on ",
    }

    dontcare_sentence = summary
    if not is_one_sentence:
        summary = summary.split('.')[0]
    res = {}

    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            if slot == "train-book people":
                start_idx -= 3

            value = re.split(
                " The | Also, | which | for | from | to | on | and | people| person", summary[start_idx:]
            )[0]

            res[slot] = value.replace(",", "").replace(".", "")

    res.update(get_dontcare_values(dontcare_sentence, "train"))

    return res


def taxi_sum_to_state(summ: str, is_one_sentence: bool) -> Dict:
    sentences = re.split("|".join(COMMON_PHRASES), summ)
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["taxi"] in sentence]
    if not summary:
        return {}
    summary = summary[0]
    slot_to_prefix = {
        "taxi-departure": " from ",
        "taxi-destination": " to ",
        "taxi-arriveby": " arrives by ",
        "taxi-leaveat": " leaves at ",
    }

    dontcare_sentence = summary
    if not is_one_sentence:
        summary = summary.split('.')[0]

    res = {}

    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]

            value = re.split(
                " The | Also, | which | from | to | and ", summary[start_idx:]
            )[0]

            res[slot] = value.replace(",", "").replace(".", "")

    res.update(get_dontcare_values(dontcare_sentence, "taxi"))

    return res


def restaurant_sum_to_state(summ: str, is_one_sentence: bool) -> dict:
    sentences = re.split("|".join(COMMON_PHRASES), summ)
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["restaurant"] in sentence]
    if not summary:
        return {}
    summary = summary[0]
    slot_to_prefix = {
        "restaurant-name": " called ",
        "restaurant-food": " serves ",
        "restaurant-area": " located in the ",
        "restaurant-pricerange": " with a",
        "restaurant-book day": " on ",
        "restaurant-book people": r" for \d+ p",
        "restaurant-book time": " at ",
    }
    res = {}

    dontcare_sentence = summary
    if not is_one_sentence:
        summary = summary.split('.')[0]

    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            if slot == "restaurant-book people":
                start_idx -= 3
            elif slot == "restaurant-pricerange":
                start_idx += 2 if summary[start_idx:].startswith("n") else 1

            _summary = summary[start_idx:]

            value = re.split(
                " The | Also, | which | called | for | on | and | at | located in the | with a| people| person| price",
                _summary,
            )[0]

            res[slot] = value.replace(",", "").replace(".", "")

    res.update(get_dontcare_values(dontcare_sentence, "restaurant"))

    return res


def attraction_sum_to_state(summ: str, is_one_sentence: bool) -> Dict:
    sentences = re.split("|".join(COMMON_PHRASES), summ)
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE['attraction'] in sentence]
    if not summary:
        return {}
    summary = summary[0]
    slot_to_prefix = {
        "attraction-name": r" called ",
        "attraction-area": r" located in the ",
        "attraction-type": r" which is a",
    }
    res = {}

    dontcare_sentence = summary
    if not is_one_sentence:
        summary = summary.split('.')[0]

    for slot, prefix in slot_to_prefix.items():
        match = re.search(prefix, summary)
        if match:
            start_idx = match.span()[-1]
            if slot == "attraction-type":
                start_idx += 2 if summary[start_idx:].startswith("n") else 1

            _summary = summary[start_idx:]

            value = re.split(
                " The | Also, | which | called | located in the ",
                _summary,
            )[0]

            value = value.replace(",", "").replace(".", "")
            res[slot] = value

    res.update(get_dontcare_values(dontcare_sentence, "attraction"))

    return res


def hotel_sum_to_state(summ: str, is_one_sentence: bool) -> dict:
    sentences = re.split("|".join(COMMON_PHRASES), summ)
    summary = [sentence for sentence in sentences if DOMAIN_PHRASE_IN_SENTENCE["hotel"] in sentence]
    if not summary:
        return {}
    summary = summary[0]
    slot_to_prefix = {
        "hotel-type": " which is a ",
        "hotel-name": " called ",
        "hotel-stars": " ranked ",
        "hotel-pricerange": " with a",
        "hotel-area": " located in the ",
        "hotel-book people": r" for \d+ p",
        "hotel-book day": " on ",
        "hotel-book stay": r" for \d+ d",
        "hotel-parking": [" has no p", " has p"],
        "hotel-internet": [" has no i", " has i"],
    }
    res = {}

    dontcare_sentence = summary
    if not is_one_sentence:
        summary = summary.split('.')[0]

    for slot, prefix in slot_to_prefix.items():
        if type(prefix) == str:
            matches = [re.search(prefix, summary)]
        else:
            matches = [re.search(p, summary) for p in prefix]
        for match in matches:
            if match:
                start_idx = match.span()[-1]
                if slot in {"hotel-book people", "hotel-book stay"}:
                    start_idx -= 3
                elif slot == "hotel-pricerange":
                    start_idx += 2 if summary[start_idx:].startswith("n") else 1

                _summary = summary[start_idx:]

                value = re.split(
                    " The | Also, | which | called | ranked | during | located in the | for | on | and | with a| people| person| price| star| day",
                    _summary,
                )[0]

                if slot in ["hotel-internet", "hotel-parking"]:
                    value = "no" if " no " in match.group() else "yes"

                res[slot] = value.replace(",", "").replace(".", "")

    res.update(get_dontcare_values(dontcare_sentence, domain="hotel"))

    return res


class MwzConverter:
    def __init__(self, wo_para: bool, do_concat: bool):
        self.domain_state_to_sum: Dict[str, Callable] = {
            "train": train_state_to_sum,
            "restaurant": restaurant_state_to_sum,
            "hotel": hotel_state_to_sum,
            "taxi": taxi_state_to_sum,
            "attraction": attraction_state_to_sum,
        }
        self.domain_sum_to_state = {
            "train": train_sum_to_state,
            "restaurant": restaurant_sum_to_state,
            "taxi": taxi_sum_to_state,
            "attraction": attraction_sum_to_state,
            "hotel": hotel_sum_to_state,
        }
        self.do_concat = do_concat
        self.wo_para = wo_para

    def sum_to_state(self, summ: str) -> Dict:
        state_dict = {}
        for _sum_to_state in self.domain_sum_to_state.values():
            state_dict.update(_sum_to_state(summ, is_one_sentence=self.do_concat))
        return state_dict

    def state_to_sum(
        self,
        ds: Dict,
        is_for_template: Optional[bool] = False,
        blank: Optional[str] = None,
    ) -> str:
        appearing_domains = list(set(k.split('-')[0] for k in ds) & set(self.domain_state_to_sum.keys()))
        random.shuffle(appearing_domains)
        either = (lambda x: blank if is_for_template else x)

        sentences = [
            self.domain_state_to_sum[domain](ds, either=either, is_one_sentence=self.do_concat, idx=idx, wo_para=self.wo_para)
            for idx, domain in enumerate(appearing_domains)
        ]
        summary = ' Also, '.join(sentences)  # when list is length 1, join does not add the Also, and just returns an str

        return summary


class DomainFreeConverter:

    def __init__(self):
        self.sentence_prefix = 'The user wants '
        self.slot_prefix = ' as '
        self.domain_prefix = ' of '
        self.phrase_divider = ', '
        self.sentence_postfix = '.'

    def state_to_sum(self, ds: Dict, is_for_template: Optional[bool] = False, blank: Optional[str] = None, is_one_sentence=True) -> str:
        """
        If we wants to generate various templates and lm ranking, we could fit better preposition for each slot
        Input:
            example: {'domain-key1': 'value1', 'key2': 'value2'}
        Returns:
            example: "The user wants key1 as value1, key2 as value2"
            real_ex: "The user wants london as departure, cambridge as destination, 12:30 as arriveby, 3 as book people,
                    tuesday as day."
        """
        res = self.sentence_prefix
        for i, (domain_slot, value) in enumerate(ds.items()):
            if i > 0:
                res += self.phrase_divider
            domain = domain_slot.split('-')[0]
            slot = domain_slot.split('-')[-1]
            phrase = value + self.slot_prefix + slot + self.domain_prefix + domain
            res += phrase

        res += self.sentence_postfix
        return res

    def sum_to_state(self, summary: str) -> Dict:
        res = {}
        summary = summary.replace(self.sentence_prefix, "")
        summary = summary.replace(self.sentence_postfix, "")
        summary = summary.split(self.phrase_divider)
        for phrase in summary:
            if self.domain_prefix not in phrase or self.slot_prefix not in phrase:
                continue
            value = phrase.split(self.slot_prefix)[0]
            slot_of_domain = phrase.split(self.slot_prefix)[-1]
            slot = slot_of_domain.split(self.domain_prefix)[0]
            domain = slot_of_domain.split(self.domain_prefix)[-1]
            res[f"{domain}-{slot}"] = value
        return res


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_converter(converter_name: str):
    # without paraphrasing
    if converter_name == "wo_para":
        return MwzConverter(wo_para=True, do_concat=True)
    # without one sentence concatenating
    if converter_name == "wo_concat":
        return MwzConverter(wo_para=False, do_concat=False)
    if converter_name == "open_domain":
        return DomainFreeConverter()
    if converter_name == "vanilla":
        return MwzConverter(wo_para=True, do_concat=False)
    return MwzConverter(wo_para=False, do_concat=True)


if __name__ == "__main__":
    sample_states = [
        ("restaurant", {
        }),
        ("restaurant", {
            "restaurant-book day": "tuesday",
            "restaurant-book people": "6",
            "restaurant-book time": "12:00",
            "restaurant-name": "meze bar",
            "restaurant-pricerange": "cheap",
            "restaurant-area": "dontcare",
            "restaurant-food": "seafood",
        }),
        ("restaurant", {
            "restaurant-name": "meze bar",
            "restaurant-pricerange": "expensive",
            "restaurant-food": "dontcare",
        }),
        ("restaurant", {
            "restaurant-name": "meze bar",
            "restaurant-pricerange": "expensive",
            "restaurant-food": "seafood",
        }),
        ("train", {
            "train-departure": "london station",
            "train-destination": "Incheon airport",
            "train-arriveby": "12:30",
            "train-book people": "3",
            "train-leaveat": "11:21",
            "train-day": "monday",
        }),
        ("train", {
            "train-departure": "dontcare",
            "train-destination": "dontcare",
            "train-arriveby": "dontcare",
            "train-book people": "3",
            "train-leaveat": "11:21",
            "train-day": "monday",
        }),
        ("hotel", {
            "hotel-type": "hotel",
            "hotel-name": "Intercontinental",
            "hotel-stars": "3",
            "hotel-pricerange": "cheap",
            "hotel-area": "east",
            "hotel-book people": "6",
            "hotel-book day": "saturday",
            "hotel-book stay": "3",
            "hotel-parking": "yes",
            "hotel-internet": "no",
        }),
        ("hotel", {
            "hotel-type": "guesthouse",
            "hotel-name": "Intercontinental",
            "hotel-parking": "no",
            "hotel-internet": "yes",
        }),
        ("hotel", {
            "hotel-type": "hotel",
            "hotel-stars": "3",
            "hotel-book people": "6",
            "hotel-book stay": "3",
            "hotel-parking": "no",
            "hotel-internet": "no",
        }),
        ("hotel", {
            "hotel-type": "guesthouse",
            "hotel-pricerange": "cheap",
            "hotel-area": "east",
            "hotel-parking": "yes",
            "hotel-internet": "yes",
        }),
        ("taxi", {
            "taxi-departure": "london station",
            "taxi-destination": "Incheon airport",
            "taxi-arriveby": "12:30",
            "taxi-leaveat": "dontcare",
        }),
        ("attraction", {
            "attraction-area": "cambridge",
            "attraction-name": "nusha",
            "attraction-type": "entertainment",
        }),
        ("restaurant-train-hotel", {
            "restaurant-book day": "tuesday",
            "restaurant-book people": "dontcare",
            "restaurant-book time": "12:00",
            "restaurant-name": "meze bar",
            "train-departure": "london station",
            "train-destination": "Incheon airport",
            "train-book people": "3",
            "train-leaveat": "dontcare",
            "train-day": "dontcare",
            "hotel-type": "guesthouse",
            "hotel-name": "Intercontinental",
            "hotel-stars": "3",
            "hotel-pricerange": "cheap",
            "hotel-area": "east",
            "hotel-book people": "6",
            "hotel-book day": "saturday",
            "hotel-book stay": "3",
            "hotel-parking": "yes",
            "hotel-internet": "no",
        }),
        ("restaurant-train-hotel", {
            "restaurant-book day": "tuesday",
            "restaurant-book people": "dontcare",
            "restaurant-book time": "12:00",
            "restaurant-name": "meze bar",
            "train-departure": "london station",
            "train-destination": "Incheon airport",
            "train-book people": "3",
            "train-leaveat": "dontcare",
            "train-day": "dontcare",
            "hotel-type": "dontcare",
            "hotel-name": "Intercontinental",
            "hotel-stars": "3",
            "hotel-pricerange": "cheap",
            "hotel-area": "east",
            "hotel-book people": "6",
            "hotel-book day": "saturday",
            "hotel-book stay": "1",
            "hotel-parking": "yes",
            "hotel-internet": "no",
        })
    ]

    for conv in ["mwz", "wo_para", "wo_concat", "open_domain"]:
        for _domain, _state in sample_states:
            converter = get_converter(conv)
            print(f'{Bcolors.HEADER}======[DOMAIN: {_domain.upper()}]======{Bcolors.ENDC}')
            print(f'Given State: {_state}')
            decoded = converter.state_to_sum(_state, is_for_template=False, blank="____")
            print(f'{Bcolors.BOLD}Decoded Summary: {decoded}{Bcolors.ENDC}')
            encoded = converter.sum_to_state(decoded)
            print(f'Restored State: {encoded}')
            print(f'{Bcolors.HEADER}Test {f"{Bcolors.OKGREEN}passed{Bcolors.ENDC}" if _state == encoded else f"{Bcolors.FAIL}failed{Bcolors.ENDC}"}!')
