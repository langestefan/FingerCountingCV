import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os

from visualization.detection import save_with_bounding_boxes


def main(config):
    logger = config.get_logger('test')
    test_batch_size = 4

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=test_batch_size,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )
    
    # test data directory
    test_data_dir = os.path.join(config['data_loader']['args']['data_dir'], 'test')
    save_dir = os.path.join('saved', 'model_output')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, targets) in enumerate(tqdm(data_loader)):

            print("len data: {}".format(len(data)))

            # specific to faster RCNN
            images = list(image.to(device) for image in data)

            targets = [{k: v.to(device)
                        for k, v in t.items()} for t in targets]

            # will return dict with boxes, labels, scores
            output = model(images, targets)
            loss = 0 # for now we don't use loss in evaluation 

            # move to cpu
            # output = output.cpu()

            for idx, image in enumerate(images):
                print(idx)
                print(image.shape)
                print("image max: {}".format(torch.max(image)))
                print("type: {}".format(type(image)))

                # take only the first bounding box with the highest score
                box = output[idx]['boxes'][0].cpu()
                label = output[idx]['labels'][0].cpu()
                score = output[idx]['scores'][0].cpu()
                print(box, label, score)

                # draw bounding box on image and store it
                file_name = os.path.join(save_dir, '{}.png'.format(idx + i*test_batch_size))
                print(file_name)
                save_with_bounding_boxes(data[idx], box, score, label, save_dir=file_name)


    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
