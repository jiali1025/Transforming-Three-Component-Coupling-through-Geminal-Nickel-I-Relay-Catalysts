import torch, pprint
ckpt_path = "/sharefs/lijl/xingjie/Ni_GAC_models/run0805/checkpoints/final_full.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")

es_state = None
for name, blob in ckpt.get("callbacks", {}).items():
    if "earlystopping" in name.lower():
        es_state = blob.get("state_dict", blob)   # Note: see the surrounding code for details.
        break

if es_state is None:
    print('Status message.')
else:
    pprint.pprint(es_state)        # Note: see the surrounding code for details.
    if es_state["stopped_epoch"] >= 0:
        print('Status message.',
              es_state["stopped_epoch"], ")")
    else:
        print('Status message.')
