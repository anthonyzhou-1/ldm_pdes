import argparse
from modules.utils import get_yaml
from dataset.datamodule import FluidsDataModule
from modules.models.ddpm import LatentDiffusion
from deepspeed.profiling.flops_profiler import FlopsProfiler

def main(args):
    config=get_yaml(args.config)
    load_dir = config['load_dir']
    trainconfig = config['training']
    dataconfig = config['data']
    modelconfig = config['model']
    dataconfig['batch_size'] = 1 # set batch size to 1 for FLOPs calculation

    datamodule = FluidsDataModule(dataconfig)

    if "model_name" in config.keys():
        model_name = config["model_name"]
        if model_name == "gnn":
            from modules.models.baselines.gnn import GNNModule
            model = GNNModule(modelconfig=modelconfig,
                    trainconfig=trainconfig,
                    normalizer=datamodule.normalizer,
                    batch_size=dataconfig["batch_size"],
                    accumulation_steps=trainconfig["accumulate_grad_batches"],)
        elif model_name == "oformer":
            from modules.models.baselines.oformer.oformer import OFormerModule
            model = OFormerModule(modelconfig=modelconfig,
                    trainconfig=trainconfig,
                    normalizer=datamodule.normalizer,
                    batch_size=dataconfig["batch_size"],
                    accumulation_steps=trainconfig["accumulate_grad_batches"],)
        elif model_name == "gino":
            from modules.models.baselines.gino import GINOModule
            model = GINOModule(modelconfig=modelconfig['gino'],
                            trainconfig=trainconfig,
                            latent_grid_size=config["model"]["latent_grid_size"],
                            normalizer=datamodule.normalizer,
                            batch_size=dataconfig["batch_size"],
                            accumulation_steps=trainconfig["accumulate_grad_batches"],)
        elif model_name == "acdm":
            from modules.models.baselines.acdm import ACDM
            modelconfig['scheduler_config'] = None
            model = ACDM(**modelconfig,
                        normalizer=datamodule.normalizer,)
        elif model_name == "unet" or model_name == "fno" or model_name == "dil_resnet":
            from modules.models.baselines.ns2D import NS2DModule
            model = NS2DModule(modelconfig=modelconfig,
                       trainconfig=trainconfig,
                       normalizer=datamodule.normalizer,
                       batch_size=dataconfig["batch_size"],
                       accumulation_steps=trainconfig["accumulate_grad_batches"],)
    else: # assume default is ldm
        modelconfig['scheduler_config'] = None
        model = LatentDiffusion(**modelconfig,
                                normalizer=datamodule.normalizer,
                                use_embed=dataconfig["dataset"]["use_embed"])
    
    model = model.cuda()

    data_loader = datamodule.train_dataloader()

    profile_step = 1
    print_profile= True

    prof = FlopsProfiler(model)

    for step, batch in enumerate(data_loader):
        for key in batch:
            if key != 'prompt':
                batch[key] = batch[key].cuda()

        # start profiling at training step "profile_step"
        if step == profile_step:
            prof.start_profile()

        loss = model.training_step(batch, step)
        # runs backpropagation
        #loss.backward()

        if step == profile_step: # if using multi nodes, check global_rank == 0 as well
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            if print_profile:
                prof.print_model_profile(profile_step=profile_step, output_file=load_dir + "flops_profile.txt")
            prof.end_profile()
            print(f"Total FLOPs: {flops}")

            exit()

        # optimizer step
        #model.optimizers().step()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepspeed FLOPs Profiler')
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    main(args)