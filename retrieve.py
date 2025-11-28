import torch
from typing import Dict

from models import load_model
from data import prepare_flickr8k_dataset
from utils import embed_images, embed_texts
from utils import _device_is_cuda, set_model_device
from gallery import visualize_topk_images

def text2image(
    model, processor, dataset, device, query_texts, top_k: int = 5,
    eval_image_bs: int = 128, eval_text_bs: int = 1, is_siglip: bool = False,
    stream_if_products_over: float = 2e8,  # switch to streaming if N*M > 2e8
) -> Dict[str, float]:
    # load image set
    images = []
    for i in range(len(dataset)):
        example = dataset[i]
        images.append(example["image"])
    # embedding
    img_emb = embed_images(model, processor, images, device, batch_size=eval_image_bs)  # (N, D)
    txt_emb = embed_texts(model, processor, query_texts, device, batch_size=eval_text_bs, is_siglip=is_siglip)  # (M, D)
    N, _ = img_emb.shape
    M, _ = txt_emb.shape

    # Use fp16 on GPU for speed/memory; ranking is unaffected by precision
    if _device_is_cuda(device):
        img_emb = img_emb.half()
        txt_emb = txt_emb.half()

    total_products = N * M
    assert total_products <= stream_if_products_over, "use small scale data!"
    # -------- Small enough: compute full matrices --------
    # sim_i2t = img_emb @ txt_emb.t()
    sim_t2i = txt_emb @ img_emb.t()
    topk_indices = torch.topk(sim_t2i, k=top_k, dim=1).indices
    return topk_indices
 
def image2text(
    model, processor, dataset, device, query_images, top_k: int = 5,
    eval_image_bs: int = 1, eval_text_bs: int = 256, is_siglip: bool = False,
    stream_if_products_over: float = 2e8,  # switch to streaming if N*M > 2e8
) -> Dict[str, float]:
    # load image set
    captions = []
    for i in range(len(dataset)):
        example = dataset[i]
        captions.append(example["image"])
    # embedding
    img_emb = embed_images(model, processor, query_images, device, batch_size=eval_image_bs)  # (N, D)
    txt_emb = embed_texts(model, processor, captions, device, batch_size=eval_text_bs, is_siglip=is_siglip)  # (M, D)
    N, _ = img_emb.shape
    M, _ = txt_emb.shape

    # Use fp16 on GPU for speed/memory; ranking is unaffected by precision
    if _device_is_cuda(device):
        img_emb = img_emb.half()
        txt_emb = txt_emb.half()

    total_products = N * M
    assert total_products <= stream_if_products_over, "use small scale data!"
    # -------- Small enough: compute full matrices --------
    sim_i2t = img_emb @ txt_emb.t()
    # sim_t2i = txt_emb @ img_emb.t()
    topk_indices = torch.topk(sim_i2t, k=top_k, dim=1).indices
    return topk_indices.detach().numpy()

if __name__ == '__main__':
    # load model & processor
    checkpoint = ''
    model, processor = load_model(checkpoint)
    # load dataset
    flickr8k = prepare_flickr8k_dataset()
    flickr8k["test"] = flickr8k["test"].shuffle(seed=5489)
    # set device
    device = set_model_device()
    model.to(device).eval()
    # retrieve
    demo_query_texts = flickr8k["test"][0]['captions']
    image_indices = text2image(model, 
                               processor, 
                               flickr8k['test'], 
                               device, 
                               query_texts=demo_query_texts, 
                               top_k=5)
    print(image_indices)
    print(demo_query_texts)
    for i in range(len(demo_query_texts)):
        visualize_topk_images(flickr8k["test"], image_indices[i,], top_k=5, savefig=f'topk-{i}.png')

