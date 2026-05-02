"""

run_attacks.py

Runs embedding-only model inversion on Version A and Version B.



Must be run AFTER both models are trained:

  python experiments/train_fl_no_dp.py

  python experiments/train_fl_with_dp.py



Usage:

  python experiments/run_attacks.py

  python experiments/run_attacks.py --client client_01



Preprocess two raw subjects (see scripts/setup_attack_raw_subjects.py) before using client_01.

"""



import sys

import os

import argparse



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from config import (

    MODEL_NO_DP,

    MODEL_WITH_DP,

    PLOTS_DIR,

    METRICS_DIR,

    CLIENTS_DIR,

    ATTACK_ITERATIONS,

    ATTACK_LR,

    STYLEGAN_NETWORK_PKL,

    ATTACK_EVAL_CLIENT_IDS,

)

from src.attacks.model_inversion import attack_both_models





def resolve_client_dir(clients_root: str, cid_raw: str) -> str:

    s = cid_raw.strip()

    if s.startswith("client_"):

        name = s

    elif s.isdigit():

        name = f"client_{int(s):02d}"

    else:

        name = f"client_{s}"

    return os.path.join(clients_root, name)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--client",

        action="append",

        dest="clients",

        help="Client folder under CLIENTS_DIR, e.g. client_01 (repeatable). Default: ATTACK_EVAL_CLIENT_IDS.",

    )

    args = parser.parse_args()



    os.makedirs(PLOTS_DIR, exist_ok=True)

    os.makedirs(METRICS_DIR, exist_ok=True)



    for path in [MODEL_NO_DP, MODEL_WITH_DP]:

        if not os.path.exists(path):

            raise FileNotFoundError(

                f"Model not found: {path}\n"

                "Run training scripts first:\n"

                "  python experiments/train_fl_no_dp.py\n"

                "  python experiments/train_fl_with_dp.py"

            )



    client_ids = args.clients if args.clients else list(ATTACK_EVAL_CLIENT_IDS)



    print("=" * 60)

    print("Model inversion attack (embedding-only objective)")

    print("=" * 60)

    if STYLEGAN_NETWORK_PKL and str(STYLEGAN_NETWORK_PKL).strip():

        print(f"Using StyleGAN: {STYLEGAN_NETWORK_PKL}")

        if str(STYLEGAN_NETWORK_PKL).lower().startswith("http"):

            print(

                "(First run may download ~364 MB; dnnlib caches the pickle.)"

            )

    else:

        print(

            "StyleGAN disabled (STYLEGAN_NETWORK_PKL empty/none): pixel-space inversion "

            "(CROPPED_DIR prior skips the victim client folder)."

        )



    for cid in client_ids:

        client_dir = resolve_client_dir(CLIENTS_DIR, cid)

        if not os.path.isdir(client_dir):

            print(f"[skip] Missing client folder: {client_dir}")

            continue



        print("\n--- Target:", client_dir, "---")

        res = attack_both_models(

            model_no_dp_path=MODEL_NO_DP,

            model_with_dp_path=MODEL_WITH_DP,

            client_dir=client_dir,

            output_dir=PLOTS_DIR,

            iterations=ATTACK_ITERATIONS,

            attack_lr=ATTACK_LR,

            plot_file_tag=os.path.basename(client_dir),

        )

        tag = res["client_tag"]

        print(

            f"Inversion losses ({tag})  "

            f"No-DP: {res['no_dp_final_loss']:.4f}  With-DP: {res['with_dp_final_loss']:.4f}"

        )

        print(

            f"  Comparison figure: {os.path.join(PLOTS_DIR, f'attack_comparison_{tag}.png')}"

        )



    print("\n" + "=" * 60)

    print("Plots under:", PLOTS_DIR)

    print("Naming: attack_comparison_<client>.png")

    print("=" * 60)





if __name__ == "__main__":

    main()


