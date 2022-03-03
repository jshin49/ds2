import json
# Strict match evaluation from https://github.com/jasonwu0731/trade-dst/blob/master/models/TRADE.py
# check utils/prediction_sample.json for the format of predictions
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def evaluate_metrics(all_prediction, SLOT_LIST):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for idx, dial in all_prediction.items():
        for k, cv in dial["turns"].items():
            if set(cv["turn_belief"]) == set(cv["pred_belief"]):
                joint_acc += 1
            else:
                print(cv["turn_belief"])
                print(cv["pred_belief"])
                print("==================")
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv["pred_belief"]), SLOT_LIST)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv["pred_belief"]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def get_acc(
    pred_slot_values: List[Dict],
    correct_slot_values: List[Dict],
    eval_slots: List[Dict],
    hide_metrics: Optional[List[str]] = None
):
    domains_per_sample = [set(k.split("-")[0] for k in x) for x in eval_slots]
    all_domains = set.union(*domains_per_sample)

    # res = defaultdict(float) # causes error in pl>=1.4.6 during teardown
    res = { 
        "jga": 0.0, "slot_acc": 0.0, "slot_none_acc": 0.0, "slot_true_acc": 0.0,
    }
    for _domain in all_domains:
        res[f"{_domain}_jga"] = 0.0
        res[f"{_domain}_slot_acc"] = 0.0
        res[f"{_domain}_slot_none_acc"] = 0.0
        res[f"{_domain}_slot_true_acc"] = 0.0

    counts = defaultdict(int)

    for _pred, _correct, _eval_slots, _domains in zip(pred_slot_values, correct_slot_values, eval_slots, domains_per_sample):
        res["jga"] += (
            set(
                (k, v) for k, v in _pred.items() if k in _eval_slots
            ) == set(
                (k, v) for k, v in _correct.items() if k in _eval_slots
            )
        )
        res["slot_acc"] += sum(
            (k in _pred and k in _correct and _pred[k] == _correct[k]) or (k not in _pred and k not in _correct)
            for k in _eval_slots
        )
        res[f"slot_true_acc"] += sum(
            (k in _pred and k in _correct and _pred[k] == _correct[k])
            for k in _eval_slots
        )
        res[f"slot_none_acc"] += sum(
            (k not in _pred and k not in _correct)
            for k in _eval_slots
        )
        num_slots = len(_eval_slots)
        num_true = len(set(_eval_slots) & set(_correct.keys()))
        counts["turn"] += 1
        counts["slot"] += num_slots
        counts["slot_true"] += num_true
        counts["slot_none"] += num_slots - num_true

        for _domain in _domains:
            domain_eval_slots = set(k for k in _eval_slots if k.split("-")[0] == _domain)
            res[f"{_domain}_jga"] += (
                set(
                    (k, v) for k, v in _pred.items() if k in domain_eval_slots
                ) == set(
                    (k, v) for k, v in _correct.items() if k in domain_eval_slots
                )
            )
            res[f"{_domain}_slot_acc"] += sum(
                (k in _pred and k in _correct and _pred[k] == _correct[k]) or (k not in _pred and k not in _correct)
                for k in domain_eval_slots
            )
            res[f"{_domain}_slot_true_acc"] += sum(
                (k in _pred and k in _correct and _pred[k] == _correct[k])
                for k in domain_eval_slots
            )
            res[f"{_domain}_slot_none_acc"] += sum(
                (k not in _pred and k not in _correct)
                for k in domain_eval_slots
            )
            num_slots = len(domain_eval_slots)
            num_true = len(set(domain_eval_slots) & set(_correct.keys()))
            counts[f"{_domain}_turn"] += 1
            counts[f"{_domain}_slot"] += num_slots
            counts[f"{_domain}_slot_true"] += num_true
            counts[f"{_domain}_slot_none"] += num_slots - num_true

    def safe_devision(x: float, y: int):
        return x / y if y > 0 else 0

    res["jga"] = safe_devision(res["jga"], counts["turn"])
    res["slot_acc"] = safe_devision(res["slot_acc"], counts["slot"])
    res["slot_true_acc"] = safe_devision(res["slot_true_acc"], counts["slot_true"])
    res["slot_none_acc"] = safe_devision(res["slot_none_acc"], counts["slot_none"])

    for domain in all_domains:
        res[f"{domain}_jga"] = safe_devision(res[f"{domain}_jga"], counts[f"{domain}_turn"])
        res[f"{domain}_slot_acc"] = safe_devision(res[f"{domain}_slot_acc"], counts[f"{domain}_slot"])
        res[f"{domain}_slot_true_acc"] = safe_devision(res[f"{domain}_slot_true_acc"], counts[f"{domain}_slot_true"])
        res[f"{domain}_slot_none_acc"] = safe_devision(res[f"{domain}_slot_none_acc"], counts[f"{domain}_slot_none"])

    if hide_metrics:
        res = {k: v for k, v in res.items() if k not in hide_metrics}
    return res


# def get_entity_recall(pred_summary: List[str], correct_slot_values: List[Dict], eval_slots):
#     slot_count = {"dontcare": 0, "proper": 0}
#     hit_count = {"dontcare": 0, "proper": 0}
#     for _sum, _slot_values in zip(pred_summary, correct_slot_values):
#         for k, v in _slot_values.items():
#             if v == "dontcare":
#                 slot_count["dontcare"] += 1
#                 dontcare_match = re.search("does not care about", _sum)
#                 if dontcare_match and k.replace(f"{domain}-", "") in _sum[dontcare_match.span()[1]:]:
#                     hit_count["dontcare"] += 1
#             else:
#                 slot_count["proper"] += 1
#                 if re.search(v, _sum) is not None:
#                     hit_count["proper"] += 1
#     num_total_slots = sum(slot_count.values())
#     entity_recall = sum(hit_count.values()) / num_total_slots if num_total_slots > 0 else 0.0
#     return entity_recall, slot_count, hit_count


def get_template_acc(
    pred_summary: List[str],
    gold_templates: List[str],
    blank: str = "____",
):
    template_correctness = [
        all(pattern in _sum for pattern in _template.split(blank))
        for _sum, _template in zip(pred_summary, gold_templates)
    ]

    return sum(template_correctness) / len(template_correctness)


if __name__ == "__main__":
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    with open("save/t5/results/zeroshot_prediction.json") as f:
        prediction = json.load(f)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(prediction, ontology)

    evaluation_metrics = {"Joint Acc": joint_acc_score, "Turn Acc": turn_acc_score, "Joint F1": F1_score}
    print(evaluation_metrics)
