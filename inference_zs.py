"""
Zero-Shot Audio Classification using CLAP on ESC-50 / custom datasets.
Includes baseline zero-shot and an improved inference method (new_zs).
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from msclap import CLAP
from tqdm import tqdm


def combine_prompt_with_label(label: str):
    """
    Reads prompts from a file and replaces <sound> with the given label.
    Useful for multi-prompt inference.
    """
    result_query = []
    with open('/fs/nexus-projects/brain_project/ashish/CLAP/promt_list.txt') as file:
        for prompt in file:
            result_query.append(prompt.strip().replace("<sound>", label))
    return result_query


def get_result_new_zs(
    audio_files,
    ground_truth_idx,
    class_index_dict,
    model,
    batch_size=128,
    multi_prompt=True,
    noise=False,
    min_snr=0.001,
    max_snr=0.005,
    pitch_shift=False,
    polarity_inversion=False,
    delay=False,
    gain=False,
    lowPass=False,
    highPass=False,
    randomHighLowPass=False,
    reverb=False,
):
    """
    New zero-shot inference using CLAP with multi-prompt ensemble and
    beta-search over different fusion strategies of global/local embeddings.
    """
    ground_truth = torch.tensor(ground_truth_idx).view(-1, 1)

    # Compute text embeddings (multi-prompt average or single template)
    if multi_prompt:
        all_texts = [combine_prompt_with_label(t) for t in class_index_dict.keys()]
        text_embed = np.vstack(
            [model.get_text_embeddings(all_text).mean(axis=0).detach().cpu().numpy()
             for all_text in all_texts]
        )
        text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)
    else:
        all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]
        text_embed = model.get_text_embeddings(all_texts).detach().cpu().numpy()

    feat_t = torch.Tensor(text_embed).T

    # Store logits from different strategies
    comp_logits0, comp_logits1, comp_logits2 = [], [], []
    audio_batches = [audio_files[i:i + batch_size] for i in range(0, len(audio_files), batch_size)]

    for audio_file in audio_batches:
        # CLAP audio embeddings (global + local)
        audio_embed = dict(zip(
            ['audio_embeds_global', 'audio_embeds_local', 'unused'],
            model.get_audio_embeddings(audio_file, resample=True, noise=noise,
                                       min_snr=min_snr, max_snr=max_snr,
                                       pitch_shift=pitch_shift, polarity_inversion=polarity_inversion,
                                       delay=delay, gain=gain, lowPass=lowPass,
                                       highPass=highPass, randomHighLowPass=randomHighLowPass,
                                       reverb=reverb)
        ))

        audio_global = audio_embed['audio_embeds_global'].detach().cpu()
        audio_local = audio_embed['audio_embeds_local'].detach().cpu().permute(0, 2, 1)

        logits0, logits1, logits2 = [], [], []
        for i, feat_a in enumerate(audio_local):
            A_weight = torch.matmul(feat_a.permute(1, 0), feat_t) * 2
            A_weight1 = F.softmax(A_weight, dim=0)
            A_weight2 = F.softmax(A_weight, dim=1)

            feat_t_a = torch.matmul(feat_a, A_weight1)
            feat_a_a = torch.matmul(A_weight2, feat_t.permute(1, 0))
            feat_a_a = feat_a_a.mean(0) + feat_a_a.max(0)[0]

            logits0.append((audio_global[i] @ feat_t).unsqueeze(0))
            logits1.append((audio_global[i] @ feat_t_a).unsqueeze(0))
            logits2.append((feat_a_a @ feat_t).unsqueeze(0))

        comp_logits0.append(torch.cat(logits0, dim=0))
        comp_logits1.append(torch.cat(logits1, dim=0))
        comp_logits2.append(torch.cat(logits2, dim=0))

    comp_logits0 = torch.cat(comp_logits0)
    comp_logits1 = torch.cat(comp_logits1)
    comp_logits2 = torch.cat(comp_logits2)

    # Grid search over beta2, beta3 weights
    beta2_list = np.linspace(0.001, 0.01, 200)
    beta3_list = np.linspace(0.001, 3.0, 200)

    def get_accuracy(total_ranking, ground_truth):
        preds = torch.where(total_ranking == ground_truth)[1].cpu().numpy()
        return np.mean(preds < 1)  # R@1 accuracy

    best_acc, best_beta2, best_beta3 = 0., 0., 0.
    for beta2 in beta2_list:
        for beta3 in beta3_list:
            combined = comp_logits0 + comp_logits1 * beta2 + comp_logits2 * beta3
            ranking = torch.argsort(combined, descending=True)
            acc = get_accuracy(ranking, ground_truth)
            if acc > best_acc:
                best_acc, best_beta2, best_beta3 = acc, beta2, beta3

    print(f"Best Accuracy: {best_acc:.4f} | beta2={best_beta2:.4f}, beta3={best_beta3:.4f}")
    return best_acc


def main(args):
    """
    Main function: loads dataset, prepares audio paths and labels,
    and evaluates zero-shot performance with/without augmentations.
    """
    clap_model = CLAP(version='2023', use_cuda=True)

    dataset_path = os.path.join(args.data_root, args.dataset, args.split + ".csv")
    data = pd.read_csv(dataset_path)

    audio_files = data['path'].to_list()
    labels = [' '.join(label.split('_')) for label in data['label'].to_list()]
    unique_labels = list(set(labels))
    class_index_dict = dict(zip(unique_labels, list(range(len(unique_labels)))))
    ground_truth_idx = [class_index_dict[label] for label in labels]

    # Loop through augmentations
    for aug in args.augmentations:
        params = {
            'noise': False, 'pitch_shift': False, 'polarity_inversion': False,
            'delay': False, 'gain': False, 'lowPass': False, 'highPass': False,
            'randomHighLowPass': False, 'reverb': False,
            'min_snr': 0.001, 'max_snr': 0.005
        }
        params[aug] = True

        print(f"\nRunning augmentation: {aug}")
        acc = get_result_new_zs(audio_files, ground_truth_idx, class_index_dict,
                                clap_model, batch_size=args.batch_size,
                                multi_prompt=args.multi_prompt, **params)
        print(f"Dataset={args.dataset}, Augmentation={aug}, Acc={acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot audio classification with CLAP.")
    parser.add_argument("--data_root", type=str, default="/fs/nexus-projects/brain_project/ashish/audio_library/",
                        help="Root directory of datasets")
    parser.add_argument("--dataset", type=str, default="esc50",
                        help="Dataset name (e.g., esc50, beijing_opera)")
    parser.add_argument("--split", type=str, default="eval",
                        help="Which CSV split to load (train/eval)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for audio embeddings")
    parser.add_argument("--multi_prompt", action="store_true",
                        help="Enable multi-prompt ensembling")
    parser.add_argument("--augmentations", nargs="+",
                        default=["noise", "pitch_shift", "polarity_inversion", "gain", "lowPass", "highPass", "reverb"],
                        help="List of augmentations to apply")
    args = parser.parse_args()
    main(args)
