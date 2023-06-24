#ifndef __BDPM_H__
#define __BDPM_H__

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/common.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/particleproc.h>
#include <mitsuba/render/photonmap.h>

//simular to bdpt/bdpt.h

MTS_NAMESPACE_BEGIN

void swapv(Vector &a, Vector &b);

struct BDPMConfiguration {
    size_t spp;
    int maxDepth, blockSize, borderSize;
    bool lightImage;
    bool sampleDirect;
    bool showWeighted;
    size_t sampleCount;
    Vector2i cropSize;
    int rrDepth;
    size_t maxPhoton;
    size_t minPhoton;
    float maxRadius;
    size_t granularity;
    size_t numShot;
    size_t totalPhotons;

    inline BDPMConfiguration(){}

    inline BDPMConfiguration(Stream *stream)
    {
        spp = stream->readSize();
        maxDepth = stream->readInt();
        blockSize = stream->readInt();
        lightImage = stream->readBool();
        sampleDirect = stream->readBool();
        showWeighted = stream->readBool();
        sampleCount = stream->readSize();
        cropSize = Vector2i(stream);
        rrDepth = stream->readInt();
        maxPhoton = stream->readSize();
        minPhoton = stream->readSize();
        maxRadius = stream->readFloat();
        granularity = stream->readSize();
    }

    inline void serialize(Stream *stream) const
    {
        stream->writeSize(spp);
        stream->writeInt(maxDepth);
        stream->writeInt(blockSize);
        stream->writeBool(lightImage);
        stream->writeBool(sampleDirect);
        stream->writeBool(showWeighted);
        stream->writeSize(sampleCount);
        cropSize.serialize(stream);
        stream->writeInt(rrDepth);
        stream->writeSize(maxPhoton);
        stream->writeSize(minPhoton);
        stream->writeFloat(maxRadius);
        stream->writeSize(granularity);
    }

    void dump() const
    {
        SLog(EDebug, "Bidirectional path tracer configuration:");
        SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
        SLog(EDebug, "   Image size                  : %ix%i",
            cropSize.x, cropSize.y);
        SLog(EDebug, "   Direct sampling strategies  : %s",
            sampleDirect ? "yes" : "no");
        SLog(EDebug, "   Generate light image        : %s",
            lightImage ? "yes" : "no");
        SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
        SLog(EDebug, "   Block size                  : %i", blockSize);
        SLog(EDebug, "   Number of samples           : " SIZE_T_FMT, sampleCount);
        SLog(EDebug, "   Max NN Photons              : ",SIZE_T_FMT, maxPhoton);
        SLog(EDebug, "   Max NN Radius               : %f", maxRadius);
        #if BDPM_DEBUG == 1
            SLog(EDebug, "   Show weighted contributions : %s", showWeighted ? "yes" : "no");
        #endif
    }
    //MTS_DECLARE_CLASS()
};

MTS_NAMESPACE_END
#endif