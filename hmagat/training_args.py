import argparse


def add_training_args(parser):
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--num_training_oe", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--imitation_learning_model", type=str, default="MAGAT")
    parser.add_argument("--cnn_mode", type=str, default="basic-CNN")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_attention_heads", type=int, default=1)
    parser.add_argument("--edge_dim", type=int, default=None)
    parser.add_argument("--model_residuals", type=str, default=None)
    parser.add_argument(
        "--use_edge_weights", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--use_edge_attr", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--hyperedge_feature_generator", type=str, default="gcn")

    parser.add_argument("--load_partial_parameters_path", type=str, default=None)
    parser.add_argument("--replace_model", type=str, default=None)
    parser.add_argument("--parameters_to_load", type=str, default="all")
    parser.add_argument("--parameters_to_freeze", type=str, default=None)

    parser.add_argument("--lr_start", type=float, default=1e-3)
    parser.add_argument("--lr_end", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="cosine-annealing")
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_validation_batches", type=int, default=None)

    parser.add_argument("--grad_clip_value", type=float, default=None)
    parser.add_argument("--grad_clip_norm", type=str, default="2.0")
    parser.add_argument("--grad_clip_type", type=str, default="norm")
    parser.add_argument("--grad_clip_warmup_steps", type=int, default=None)

    parser.add_argument("--validation_every_epochs", type=int, default=4)
    parser.add_argument(
        "--run_online_expert", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--save_intmd_checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")

    parser.add_argument(
        "--skip_validation", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--skip_validation_accuracy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--initial_val_size", type=int, default=128)
    parser.add_argument("--threshold_val_success_rate", type=float, default=0.9)
    parser.add_argument("--num_run_oe", type=int, default=500)
    parser.add_argument("--run_oe_after", type=int, default=0)
    parser.add_argument(
        "--recursive_oe", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument(
        "--load_positions_separately",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_on_terminated_agents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run_expert_in_separate_fork",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--use_target_vec", type=str, default=None)
    parser.add_argument("--collision_shielding", type=str, default="naive")
    parser.add_argument("--action_sampling", type=str, default="deterministic")
    parser.add_argument("--action_sampling_temperature", type=float, default=1.0)
    parser.add_argument(
        "--action_sampling_temperature_strategy", type=str, default="constant"
    )
    parser.add_argument("--cs_max_dist_to_goal", type=int, default=2)
    parser.add_argument("--cs_agent_radius", type=int, default=3)
    parser.add_argument("--cs_num_steps", type=int, default=3)
    parser.add_argument("--cs_move_from_goal_threshold", type=float, default=0.0)

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_attr_opts", type=str, default="straight")
    parser.add_argument("--pre_gnn_embedding_size", type=int, default=None)
    parser.add_argument("--pre_gnn_num_mlp_layers", type=int, default=None)

    parser.add_argument("--intmd_training", type=str, default=None)
    parser.add_argument(
        "--pass_cnn_output_to_gnn2",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--collision_shielding_args", type=str, default="")
    parser.add_argument("--collision_shielding_model_epoch_num", type=str, default=None)

    parser.add_argument("--module_residual", type=str, default=None)

    parser.add_argument(
        "--lin_x_before_additional_data",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--use_edge_attr_for_messages", type=str, default=None)
    parser.add_argument("--edge_attr_cnn_mode", type=str, default=None)

    parser.add_argument("--max_runtime_oe", type=float, default=None)

    parser.add_argument(
        "--legacy_agents", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--final_feature_generator", type=str, default="magat")

    parser.add_argument(
        "--coordination_state_size",
        type=int,
        default=0,
        help=(
            "Hidden size for an optional recurrent coordination-state token. "
            "Set to 0 to disable it."
        ),
    )
    parser.add_argument(
        "--coordination_state_update",
        type=str,
        default="gru",
        choices=["gru"],
        help="Update rule for the optional coordination-state token.",
    )
    parser.add_argument(
        "--sequence_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train over episode sequences instead of independent snapshots.",
    )
    parser.add_argument(
        "--sequence_detach_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Detach coordination state between timesteps in sequence training. "
            "Leave disabled for full BPTT inside a sequence batch."
        ),
    )
    parser.add_argument(
        "--truncated_bptt_length",
        type=int,
        default=None,
        help=(
            "Detach coordination state every N timesteps in sequence training. "
            "Use None for full BPTT over the whole sequence batch."
        ),
    )
    parser.add_argument(
        "--validate_sequence_training_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run sequence dataset assumption checks before sequence training. "
            "Disable only after an explicit dataset audit has already passed."
        ),
    )

    parser.add_argument(
        "--oe_improve_quality", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--oe_improve_quality_threshold", type=float, default=0.8)
    parser.add_argument("--oe_improve_quality_period", type=int, default=16)
    parser.add_argument("--oe_improve_quality_buffer", type=float, default=1.2)
    parser.add_argument("--oe_improve_quality_max_num", type=int, default=30)
    parser.add_argument("--oe_improve_quality_expert", type=str, default=None)

    parser.add_argument("--pretrain_weights_path", type=str, default=None)

    return parser
