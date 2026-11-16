import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from qubic.lib.scanning_strategy.observation_plan.observationCampaign import ObservationCampaign

def main():

    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1]).expanduser().resolve()
    else:
        config_file = Path(__file__).parent / "configs" / "conf.toml"

    if not config_file.exists() or config_file.suffix != ".toml":
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")

    campaign = ObservationCampaign(config_file)
    campaign.analyze_observability()
    campaign.plan_observations()


if __name__ == '__main__':
    main()
