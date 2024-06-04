zero shot speaker adaptation (base, GST)
========================================

    /* encoder_depth=2, decoder_n_blocks=3, decoder_block_depth=3, reduction=1,
       encoder_n_head=2, embed_dim=128, encoder_kernel_size=5,
       decoder_kernel_size=5, encoder_expansion=2, punct_embed_dim=16,
       gst_n_style_tokens=500, gst_n_heads=8, gst_ref_enc_filters=[32, 32, 64, 64, 128, 128] */
    wav, mel_len, duration = ZeroVox(x)
    {

        /* emb_size=144, n_mels=80, gst_n_style_tokens=500, gst_n_heads=8, gst_ref_enc_filters=[...]*/
        style_embed = GST(x["ref_mel"] [batch_size, 1415, 80])
        -> [batch_size, 1, 144]


        /* depth, reduction, head, embed_dim, kernel_size, expansion, punct_embed_dim, speaker_embed_dim */
        pred = PhonemeEncoder(x, style_embed=style_embed, train=self.training)
        {
            phoneme      [batch_size,  82]
            puncts       [batch_size,  82]
            spkembs      [batch_size, 192]
            phoneme_mask [batch_size,  82]

            /* embed_dim=128, punct_embed_dim=16,
               depth=2, kernel_size=5, expansion=2, reduction=1, head=2 */
            features, mask = Encoder (phoneme, puncts, phoneme_mask)
            {
                /* embed_dim, punct_embed_dim */
                x = Embedding(phoneme) . Embedding(puncts)

                /* depth, head, kernel_size, expansion, reduction */
                [ depth x features], decoder_mask = [depth x] attn_blocks(x, mask)

            } -> depth x [batch_size, 82, 144], [batch_size, 82, 144]

            /* dims = [144, 288] (from encoder), kernel_size=5 */
            fused_features = Fuse(features, mask)
            {
                len(dims) x { Linear, ConvTranspose1d }
            } -> [batch_size, 82, 144]

            /* append speaker embedding to each phoneme */
            spkembs_features [batch_size, 82, 192]
            fused_features = torch.cat((fused_features, spkembs_features), dim=-1) [batch_size, 82, 336]
            mask [batch_size, 82, 336]

            /* dim = ((embed_dim+punct_embed_dim) // reduction) + speaker_embed_dim = 336 */
            pitch_pred [batch_size, 82, 1]
            pitch_features    = AcousticDecoder(fused_features) [batch_size, 82, 336]
            /* dim = 336 */
            energy_features   = AcousticDecoder(fused_features) [batch_size, 82, 336]
            /* dim = 336 */
            duration_pred, duration_features = AcousticDecoder(fused_features)
                                               [batch_size, 82, 1], [batch_size, 82, 336]

            fused_features = torch.cat([fused_features, pitch_features, energy_features, duration_features])
                             [batch_size, 82, 1344]

            fused_masks [batch_size, 82, 1344]

            features, masks, mel_len = FeatureUpsampler(fused_features, fused_masks, duration_pred, max_mel_len)
            [batch_size, 666, 1344]
        }
        -> {
              'pitch'   : [batch_size, 180, 1],
              'energy'  : [batch_size, 180, 1],
              'duration': [batch_size, 180, 1],
              'mel_len' : [batch_size],
              'features': [batch_size, 1415, 576],
              'masks'   : [batch_size, 1415, 576]
           }

        /* dim=208, decoder_kernel_size=5, n_blocks=3, block_depth=3, n_mel_channels=80 */
        mel = MelDecoder (pred['features'])

        if masks is not None and mel.size(0) > 1:
            masks = masks[:, :, :mel.shape[-1]]
            mel = mel.masked_fill(masks, 0)

        mel = mel.transpose(1, 2)
        wav = HifiGAN(mel).squeeze(1)
    }


Speaker Embedding
=================

    wav, mel_len, duration = EfficientSpeech(x)
    {

        mel, mel_len, duration = Phoneme2Mel(phoneme, phoneme_mask)
        {
            /* depth, reduction, head, embed_dim, kernel_size, expansion, punct_embed_dim, speaker_embed_dim */
            pitch, energy, duration, mel_len, features, masks = PhonemeEncoder (phoneme, phoneme_mask)
            {

                /* num_speakers=10, embed_dim=128, punct_embed_dim=16, speaker_embed_dim=64,
                   depth=2, kernel_size=5, expansion=2, reduction=1, head=2 */
                features, mask = Encoder (phoneme, puncts, speakers, phoneme_mask)
                {
                    /* num_speakers, embed_dim, punct_embed_dim, speaker_embed_dim */
                    x = Embedding(phoneme) . Embedding(puncts) . Embedding(speakers)

                    /* depth, head, kernel_size, expansion, reduction */
                    [ depth x features], decoder_mask = [depth x] attn_blocks(x, mask)

                } -> depth x [1, 37, 208]

                /* dims = [208, 416] (from encoder), kernel_size=5 */
                fused_features = Fuse(features, mask)
                {
                    len(dims) x { Linear, ConvTranspose1d }
                } -> [1, 37, 208]

                dim = (embed_dim+punct_embed_dim+speaker_embed_dim) // reduction = 208

                /* dim */
                pitch_features    = AcousticDecoder(fused_features) [1, 37, 208]
                /* dim */
                energy_features   = AcousticDecoder(fused_features) [1, 37, 208]
                /* dim */
                duration_pred, duration_features = AcousticDecoder(fused_features) [1, 37, 208]

                fused_features = torch.cat([fused_features, pitch_features, energy_features, duration_features])
                                 [1, 37, 832]

                features, masks, mel_len = FeatureUpsampler(fused_features, fused_masks, duration_pred, max_mel_len)
                [1, 288, 832]
            }

            /* dim=208, decoder_kernel_size=5, n_blocks=3, block_depth=3, n_mel_channels=80 */
            mel = MelDecoder (features)

            if masks is not None and mel.size(0) > 1:
                masks = masks[:, :, :mel.shape[-1]]
                mel = mel.masked_fill(masks, 0)
        }

        mel = mel.transpose(1, 2)
        wav = HifiGAN(mel).squeeze(1)
    }

Hyperparams
===========

                           base     small      tiny
    depth                  2        2          2
    reduction              1        2          4
    head                   2        1          1
    embed_dim              128      128        128
    kernel_size            5        3          3
    expansion              2        1          1
    punct_embed_dim        16       16         16
    speaker_embed_dim      64       64         64

    decoder_kernel_size    5        5          5
    n_blocks               3        3          2
    block_depth            3        2          2

