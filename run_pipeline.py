import config
import importlib

def main():
    cfg = config.config
    pipeline_name = cfg.get("pipeline", False)
    if not pipeline_name:
        print("No pipeline specified in config, exiting")
        return
    pipeline = importlib.import_module(f"mimeta_pipelines.{pipeline_name}")

    pipeline.get_unified_data(**cfg.pipeline_args[pipeline_name])


if __name__ == "__main__":
    main()
