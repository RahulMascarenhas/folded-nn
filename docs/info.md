## How it works

8x8 binary image -> frozen ternary layer (64 -> 56, compiled into logic,
cannot be changed) -> loadable ternary head (56 -> 6, 90 bytes over SPI)
-> six signed scores. 784 cycles per inference.

Trained on EMNIST. The shipped head recognises digits 0-5. Loading a
different 90 bytes retargets it to six different characters.

## How to test

| Pin | Function |
|---|---|
| `ui[0]` / `ui[1]` | image bit / shift |
| `ui[2]` | start |
| `ui[3]` / `ui[4]` | head bit / shift |
| `uo[7:0]` | signed score |
| `uio[0]` / `uio[1]` / `uio[2]` | valid / done / busy |
| `uio[5:3]` | class index |

Reset, shift 720 head bits in MSB first, shift 64 pixels in MSB first,
pulse start, read `uo` while `uio[0]` is high until `uio[1]` goes high.

Input must be cropped, scaled and centred the way EMNIST is, or accuracy
drops sharply.

## External hardware

None.
