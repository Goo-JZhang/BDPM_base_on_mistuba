#include<mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/bidir/util.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/range.h>
#include <mitsuba/render/photon.h>
#include "bdpm.h"

MTS_NAMESPACE_BEGIN

void swapv(Vector &a, Vector &b)
{
    Vector tmp = a;
    a = b;
    b = tmp;
}

struct ProbPath
{
    std::vector<Float> posProb;
    std::vector<Float> revProb;
    inline void putProb(Float pp, Float rp)
    {
        posProb.push_back(pp);
        revProb.push_back(rp);
    }

    inline size_t size()
    {
        return posProb.size();
    }

    inline void clear()
    {
        posProb.clear();
        revProb.clear();
    }
};

class ProbPhotonVector : public WorkResult
{
public:
    inline void nextParticle()
    {
        m_particleIndices.push_back((uint32_t) m_photons.size());
    }

    inline void put(const Photon &p)
    {
        m_photons.push_back(p);
    }

    inline void putProbs(const std::vector<float> pp, const std::vector<float> rp)
    {
        m_posProbs.push_back(pp);
        m_revProbs.push_back(rp);
    }

    inline size_t size() const
    {
        return m_photons.size();
    }

    inline size_t posSize() const
    {
        return m_posProbs.size();
    }

    inline size_t revSize() const
    {
        return m_revProbs.size();
    }

    inline size_t getParticleCount() const
    {
        return m_particleIndices.size() - 1;
    }

    inline size_t getParticleIndex(size_t idx) const
    {
        return m_particleIndices.at(idx);
    }

    inline std::vector<float> getPosProb(size_t idx) const
    {
        return m_posProbs[idx];
    }

    inline std::vector<float> getRevProb(size_t idx) const
    {
        return m_revProbs[idx];
    }

    inline void clear()
    {
        m_photons.clear();
        m_particleIndices.clear();
    }

    inline const Photon &operator[](size_t index) const
    {
        return m_photons[index];
    }

    void load(Stream *stream)
    {
        clear();
        size_t count = (size_t) stream->readUInt();
        m_particleIndices.resize(count);
        stream->readUIntArray(&m_particleIndices[0], count);
        count = (size_t) stream->readUInt();
        m_photons.resize(count);
        /*
        m_posProbs.resize(count);
        m_revProbs.resize(count);

        for(size_t i = 0; i<count; ++i)
        {
            stream->readFloatArray(&m_posProbs[i][0], m_posProbs[i].size());
        }
        for(size_t i = 0; i<count; ++i)
        {
            stream->readFloatArray(&m_revProbs[i][0], m_revProbs[i].size());
        }
        */
        for(size_t i=0; i<count; ++i)
        {
            m_photons[i] = Photon(stream);
        }
    }

    void save(Stream *stream) const
    {
        stream->writeUInt((uint32_t) m_particleIndices.size());
        stream->writeUIntArray(&m_particleIndices[0], m_particleIndices.size());
        stream->writeUInt((uint32_t) m_photons.size());
        size_t count = m_photons.size();
        /*
        for(size_t i = 0; i<count; ++i)
        {
            stream->writeFloatArray(&m_posProbs[i][0], m_posProbs[i].size());
        }
        for(size_t i = 0; i<count; ++i)
        {
            stream->writeFloatArray(&m_revProbs[i][0], m_revProbs[i].size());
        }
        */
        for (size_t i=0; i<m_photons.size(); ++i)
        {
            m_photons[i].serialize(stream);
        }
    }

    std::string toString() const 
    {
        std::ostringstream oss;
        oss << "PhotonVector[size=" << m_photons.size() << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
protected:
    virtual ~ProbPhotonVector(){}

private:
    std::vector<Photon> m_photons;
    std::vector<std::vector<float>> m_posProbs;
    std::vector<std::vector<float>> m_revProbs;
    std::vector<uint32_t> m_particleIndices;
};

MTS_IMPLEMENT_CLASS(ProbPhotonVector, false, WorkResult)

class ProbPhotonWorker : public ParticleTracer
{
public:
    ProbPhotonWorker(size_t granularity,int maxDepth, int rrDepth) 
    : ParticleTracer(maxDepth, rrDepth, false),
        m_granularity(granularity) { }
    
    ProbPhotonWorker(Stream *stream, InstanceManager *manager)
    :ParticleTracer(stream, manager)
    {
        m_granularity = stream->readSize();
    }
    
    ref<WorkProcessor> clone() const
    {
        return new ProbPhotonWorker(m_granularity, m_maxDepth, m_rrDepth);
    }

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        ParticleTracer::serialize(stream, manager);
        stream->writeSize(m_granularity);
    }

    ref<WorkResult> createWorkResult() const
    {
        return new ProbPhotonVector();
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult,
                const bool &stop)
    {
        m_workResult = static_cast<ProbPhotonVector *>(workResult);
        m_workResult->clear();
        //True tracer
        photonTracer(workUnit, workResult, stop);
        //std::cout<<"generate: "<<m_workResult->size()<<","<<
        //                m_workResult->posSize()<<","<<m_workResult->revSize()<<std::endl;
        m_workResult->nextParticle();
        m_workResult = NULL;
    }

    void photonTracer(const WorkUnit *workUnit, WorkResult *workResult,
                const bool &stop)
    {
        const RangeWorkUnit *range = static_cast<const RangeWorkUnit *>(workUnit);
        MediumSamplingRecord mRec;
        Intersection its;
        ref<Sensor> sensor = m_scene->getSensor();
        bool needsTimeSample = sensor->needsTimeSample();
        PositionSamplingRecord pRec(sensor->getShutterOpen() + 0.5f * sensor->getShutterOpenTime());

        m_sampler->generate(Point2i(0));
        
        for(size_t index = range->getRangeStart(); index<= range->getRangeEnd() && !stop; ++index)
        {
            m_sampler->setSampleIndex(index);
            //std::cout<<index<<" , "<<m_sampler->getSampleCount()<<std::endl;
            
            if(needsTimeSample)
                pRec.time = sensor->sampleTime(m_sampler->next1D());

            const Emitter *emitter = NULL;
            const Medium *medium;

            Spectrum power;
            Ray ray;

            //photon from the emitter
            //power = m_scene->sampleEmitterPosition(pRec, m_sampler->next2D());
            //std::cout<<power.toString()<<std::endl;
            //emitter = static_cast<const Emitter*>(pRec.object);
            //medium = emitter->getMedium();
            //handleNewParticle();

            //DirectionSamplingRecord dRec;
            power = m_scene->sampleEmitterRay(ray, emitter,
                        m_sampler->next2D(), m_sampler->next2D(), pRec.time);
            medium = emitter->getMedium();
            //std::cout<<"emit power: " + power.toString() + "\n";
            handleNewParticle();
            //ray.setTime(pRec.time);
            //ray.setOrigin(pRec.p);
            //ray.setDirection(dRec.d);

            Spectrum throughput(1.0f);

            int depth = 1, nullInteractions = 0;
            bool delta = false;

            float added_posprob = 1.0f, added_revprob = 1.0f;
            Float dirpdf(1.0f);
            std::vector<float> revprob = {}, posprob = {};
            float q = 1.0f;
            //std::cout<<m_workResult->size()<<std::endl;
            while(!throughput.isZero() && (depth <= m_maxDepth || m_maxDepth<0))
            {
                m_scene->rayIntersectAll(ray, its);
                //std::cout<<throughput.toString()<<std::endl;
                
                if(medium && medium->sampleDistance(Ray(ray, 0, its.t), mRec, m_sampler))
                {
                    //std::cout<<"medium"<<std::endl;
                    throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                    added_posprob *= mRec.pdfSuccess * q * dirpdf;

                    added_revprob *= mRec.pdfSuccess * q;

                    handleMediumInteraction(depth, nullInteractions, 
                                delta, mRec, medium, -ray.d, throughput*power);
                    
                    PhaseFunctionSamplingRecord pRec(mRec, -ray.d, EImportance);

                    throughput *= medium->getPhaseFunction()->sample(pRec, dirpdf, m_sampler);
                    
                    delta = false;

                    ray = Ray(mRec.p, pRec.wo, ray.time);
                    ray.mint = 0;

                    //revprob
                    swapv(pRec.wi, pRec.wo);
                    added_revprob *= medium->getPhaseFunction()->pdf(pRec);
                    revprob.push_back(added_revprob);
                    posprob.push_back(added_posprob);

                    handleProbs(posprob, revprob);
                }
                else if(its.t == std::numeric_limits<Float>::infinity())
                {
                    break;
                }
                else
                {
                    if(medium)
                    {
                        throughput *= mRec.transmittance/mRec.pdfFailure;
                        added_posprob *= mRec.pdfFailure;
                        added_revprob *= mRec.pdfFailure;
                    }
                    added_posprob *= q * dirpdf;
                    added_revprob *= q;

                    const BSDF *bsdf = its.getBSDF();

                    handleSurfaceInteraction(depth, nullInteractions, delta, its, medium, throughput*power);
                    if((throughput*power).max() > 2e8) std::cout<<(throughput*power).toString()+"\n";
                    BSDFSamplingRecord bRec(its, m_sampler, EImportance);

                    Spectrum bsdfWeight = bsdf->sample(bRec, dirpdf, m_sampler->next2D());
                    if(bsdfWeight.isZero())
                    {
                        posprob.push_back(added_posprob);
                        revprob.push_back(added_revprob);
                        handleProbs(posprob, revprob);
                        break;
                    }
                    throughput *= bsdfWeight;
                    /*
                    Vector wi = -ray.d, wo = its.toWorld(bRec.wo);
                    Float wiDotGeoN = dot(its.geoFrame.n, wi),
                        woDotGeoN = dot(its.geoFrame.n, wo);
                    if (wiDotGeoN * Frame::cosTheta(bRec.wi) <= 0 ||
                        woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                    {
                        handleProbs(posprob, revprob);
                        break;
                    }
                    */

                    if(its.isMediumTransition()) medium = its.getTargetMedium(dot(its.geoFrame.n, its.toWorld(bRec.wo)));

                    if(bRec.sampledType & BSDF::ENull) ++nullInteractions;
                    else delta = bRec.sampledType & BSDF::EDelta;

                    ray.setOrigin(its.p);
                    ray.setDirection(its.toWorld(bRec.wo));
                    ray.mint = Epsilon;

                    //revprob
                    swapv(bRec.wi, bRec.wo);
                    added_revprob *= bsdf->pdf(bRec, delta ? EMeasure::EDiscrete: EMeasure::ESolidAngle);
                    revprob.push_back(added_revprob);
                    posprob.push_back(added_posprob);
                    //posprob *= dirpdf;

                    handleProbs(posprob, revprob);

                    //if(depth==1) std::cout<<added_posprob<<" , "<<throughput.toString()<<std::endl;
                }
                if(depth++ >= m_rrDepth)
                {
                    q = std::min(throughput.max(), Float(0.95f));
                    if(m_sampler->next1D() >= q) break;
                    throughput /= q;
                }
            }
            //if(depth == 1) std::cout<<"emit into vacumm\n";
        }
    }

    void handleNewParticle()
    {
        m_workResult->nextParticle();
    }

    void handleMediumInteraction(int depth, int nullInteractions,
    bool delta, const MediumSamplingRecord &mRec, const Medium *medium,
    const Vector &wi, const Spectrum &weight)
    {
        m_workResult->put(Photon(mRec.p, Normal(0.0f,0.0f,0.0f), 
                    -wi, weight, depth-nullInteractions));
    }

    void handleProbs(std::vector<float> posprob, std::vector<float> revprob)
    {
        m_workResult->putProbs(posprob, revprob);
    }

    void handleSurfaceInteraction(int depth, int nullInteractions, bool delta,
            const Intersection &its, const Medium *medium,
            const Spectrum &weight) 
    {
        m_workResult->put(Photon(its.p, its.geoFrame.n, -its.toWorld(its.wi), 
                                weight, depth - nullInteractions));
    }

    MTS_DECLARE_CLASS()

protected:
    virtual ~ProbPhotonWorker(){}
protected:
    size_t m_granularity;
    ref<ProbPhotonVector> m_workResult;
};

MTS_IMPLEMENT_CLASS_S(ProbPhotonWorker, false, ParticleTracer);

class ProbPhotonProcess : public ParticleProcess
{
public:
    ProbPhotonProcess(size_t photonCount,
    size_t granularity, int maxDepth, int rrDepth, bool isLocal, bool autoCancel,
    const void *progressReporterPayload)
    : ParticleProcess(ParticleProcess::EGather, photonCount, granularity, "Gathering Prob Photons",
    progressReporterPayload), m_photonCount(photonCount), m_maxDepth(maxDepth),
    m_rrDepth(rrDepth), m_isLocal(isLocal), m_autoCancel(autoCancel), m_excess(0),
    m_numShot(0)
    {
        std::cout<<"allocating space of "<<photonCount<<" photons"<<std::endl;
        m_photonMap = new PhotonMap(photonCount);
        m_posProbs = new std::vector<std::vector<float>>();
        m_revProbs = new std::vector<std::vector<float>>();
        //std::cout<<"initialized successful"<<std::endl;
    }

    ref<WorkProcessor> createWorkProcessor() const
    {
        return new ProbPhotonWorker(m_granularity, m_maxDepth, m_rrDepth);
    }

    void processResult(const WorkResult *wr, bool cancelled)
    {
        if(cancelled) return;

        const ProbPhotonVector &vec = *static_cast<const ProbPhotonVector *>(wr);
        LockGuard lock(m_resultMutex);

        size_t nParticles = 0;
        //std::cout<<"Particle Nums:"<<vec.getParticleCount()<<std::endl;
        for(size_t i=0; i<vec.getParticleCount(); ++i)
        {
            size_t start = vec.getParticleIndex(i),
                    end = vec.getParticleIndex(i+1);
            if(start < end) ++nParticles;
            bool full = false;
            //if(start == end) std::cout<<"invalid emit"<<std::endl;
            //std::cout<<"start = "<<start<<" , end = "<<end<<std::endl;
            for(size_t j = start; j<end; ++j)
            {
                //std::cout<<"j = "<<j<<","<<vec.size()<<","<<vec.posSize()<<","<<vec.revSize()<<std::endl;
                if(!m_photonMap->tryAppend(vec[j]))
                {
                    m_excess += vec.size() - j;
                    full = true;
                    break;
                }
                //std::cout<<vec.getRevProb(j).size()<<std::endl;
                m_posProbs->push_back(vec.getPosProb(j));
                m_revProbs->push_back(vec.getRevProb(j));
            }
            if(full) break;
        }
        m_numShot += nParticles;
        increaseResultCount(vec.size());
    }
    
    inline PhotonMap* getPhotonMap()
    {
        return m_photonMap;
    }

    inline std::vector<std::vector<float>>* getPosProbs()
    {
        return m_posProbs;
    }

    inline std::vector<std::vector<float>>* getRevProbs()
    {
        return m_revProbs;
    }

    inline size_t getExcessPhotons() const
    {
        return m_excess;
    }

    inline size_t getShotParticles() const
    {
        return m_numShot;
    }

    bool isLocal() const
    {
        return m_isLocal;
    }
    
    MTS_DECLARE_CLASS()
protected:
    virtual ~ProbPhotonProcess(){}

protected:
    ref<PhotonMap> m_photonMap;
    std::vector<std::vector<float>>* m_posProbs;
    std::vector<std::vector<float>>* m_revProbs;
    size_t m_photonCount;
    int m_maxDepth;
    int m_rrDepth;
    bool m_isLocal;
    bool m_autoCancel;
    size_t m_excess, m_numShot;
};

MTS_IMPLEMENT_CLASS(ProbPhotonProcess, false, ParticleProcess)

class BDPMWorkResult : public WorkResult
{
public:
    BDPMWorkResult(const BDPMConfiguration &config, const ReconstructionFilter *rfilter,
                    Vector2i blockSize = Vector2i(-1,-1))
    {
        if(blockSize == Vector2i(-1,-1))
            blockSize = Vector2i(config.blockSize, config.blockSize);
        
        m_block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter);
        m_block->setOffset(Point2i(0,0));
        m_block->setSize(blockSize);
        /*
        if(config.lightImage)
        {
            m_lightImage = new ImageBlock(Bitmap::ESpectrum, 
                    config.cropSize, rfilter);
            m_lightImage->setSize(config.cropSize);
            m_lightImage->setOffset(Point2i(0,0));
        }
        */
    }

    void put(const BDPMWorkResult *workResult)
    {
        m_block->put(workResult->m_block.get());
        //if(m_lightImage) m_lightImage->put(workResult->m_lightImage.get());
    }

    void clear()
    {
        //if(m_lightImage) m_lightImage->clear();
        
        m_block->clear();
    }

    void load(Stream *stream)
    {
        //if (m_lightImage) m_lightImage->load(stream);
        
        m_block->load(stream);
    }

    void save(Stream *stream) const
    {
        //if (m_lightImage.get()) m_lightImage->save(stream);

        m_block->save(stream);
    }

    std::string toString() const
    {
        return m_block->toString();
    }

    void putSample(const Point2 &sample, const Spectrum &spec)
    {
        m_block->put(sample, spec, 1.0f);
    }

    const ImageBlock *getImageBlock() const
    {
        return m_block.get();
    }

    void setSize(const Vector2i &size)
    {
        m_block->setSize(size);
    }

    void setOffset(const Point2i &offset)
    {
        m_block->setOffset(offset);
    }

    MTS_DECLARE_CLASS()
protected:
    virtual ~BDPMWorkResult(){};
    ref<ImageBlock> m_block;
};

MTS_IMPLEMENT_CLASS(BDPMWorkResult, false, WorkResult)

class BDPMRenderer : public WorkProcessor
{
public:
    BDPMRenderer(const BDPMConfiguration&config) : m_config(config){}

    BDPMRenderer(Stream *stream, InstanceManager *manager)
                :WorkProcessor(stream, manager), m_config(stream){}
    
    BDPMRenderer(const BDPMConfiguration&config, 
                const PhotonMap* photonmap, 
                const std::vector<std::vector<float>>* posp,
                const std::vector<std::vector<float>>* revp,
                size_t numShot)
    : m_config(config), m_photonMap(photonmap), m_posProbs(posp), m_revProbs(revp),
    m_numShot(numShot)
    {}

    virtual ~BDPMRenderer(){}

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        m_config.serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const
    {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult() const
    {
        return new BDPMWorkResult(m_config, m_rfilter.get(), 
                                Vector2i(m_config.blockSize));
    }

    /*
    void loadPhotons(PhotonMap* photonmap, 
                    std::vector<std::vector<float>>* posprob, 
                    std::vector<std::vector<float>>* revprob)
    {
        m_photonMap = photonmap;
        m_posProbs = posprob;
        m_revProbs = revprob;
    }
    */

    void prepare()
    {
        Scene *scene = static_cast<Scene *>(getResource("scene"));
        m_scene = new Scene(scene);
        m_sampler = static_cast<Sampler *>(getResource("sampler"));
        m_sensor = static_cast<Sensor *>(getResource("sensor"));
        m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
        m_scene->removeSensor(scene->getSensor());
        m_scene->addSensor(m_sensor);
        m_scene->setSensor(m_sensor);
        m_scene->setSampler(m_sampler);
        m_scene->wakeup(NULL, m_resources);
        m_scene->initializeBidirectional();
        m_maxRadius = m_config.maxRadius * m_scene->getBSphere().radius;
    }

    ref<WorkProcessor> clone() const
    {
        //return new BDPMRenderer(m_config);
        return new BDPMRenderer(m_config, m_photonMap, m_posProbs, m_revProbs, m_numShot);
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        /* Prepare the sampler for tile-based rendering */
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool&stop)
    {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit*>(workUnit);
        BDPMWorkResult *result = static_cast<BDPMWorkResult*>(workResult);
        bool needsTimeSample = m_sensor->needsTimeSample();
        Float stime = m_sensor->getShutterOpen();

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clear();
        m_hilberCurve.initialize(TVector2<uint8_t>(rect->getSize()));
        size_t spp = std::min(m_config.spp, m_sampler->getSampleCount());

        ProbPath sensorRayPath;

        int maxDepth = m_config.maxDepth;

        for(size_t i = 0; i<m_hilberCurve.getPointCount(); ++i)
        {
            Point2i offset = Point2i(m_hilberCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            //std::cout<<m_sampler.toString()<<std::endl;
            
            //std::cout<<m_sampler->getSampleIndex()<<" , "<<m_sampler->getSampleCount()<<std::endl;

            MediumSamplingRecord mRec;

            Spectrum L(0.0f);

            for(size_t j = 0; j< spp ; j++)
            {
                if(stop) break;

                if(needsTimeSample) stime = m_sensor->sampleTime(m_sampler->next1D());

                RayDifferential ray;
                Spectrum throughput  = m_sensor->sampleRay(ray, Point2(offset), m_sensor->needsApertureSample()? m_sampler->next2D():Point2(0.5f), stime);

                const Medium* medium = m_sensor->getMedium();

                int depth = 1;

                bool delta = true;

                float posprob = 1.0f, revprob = 1.0f, q = 1.0f;
                Float dirpdf(1.0f);

                Intersection its;

                sensorRayPath.clear();

                while(!throughput.isZero() && (depth<=maxDepth || maxDepth <0))
                //while(depth<=1)
                {
                    //L += m_scene->evalEnvironment(ray);
                    //m_scene->getEnvironmentEmitter();
                    m_scene->rayIntersect(ray, its);
                    //direct luminate
                    if(!its.isValid() && delta)
                    {
                        if(m_scene->hasEnvironmentEmitter())
                        {
                            Spectrum envlight = m_scene->evalEnvironment(ray);
                            L += throughput * envlight;
                            //if((throughput * m_scene->evalEnvironment(ray)).max() > 1)
                            //{
                            //    m_scene->evalEnvironment(ray);
                            //    std::cout<<"env light:" + throughput.toString() + " , " +  m_scene->evalEnvironment(ray).toString() + "\n";
                            //}
                        }
                        break;
                    }

                    if(depth == 1 && its.isEmitter())
                    {
                        L += throughput * its.Le(-ray.d);
                        //if((throughput * its.Le(-ray.d)).max() > 1)std::cout<<"emit light:" + (throughput * its.Le(-ray.d)).toString() + "\n";
                    }

                    if(its.hasSubsurface())
                    {
                        L += throughput * its.LoSub(m_scene, m_sampler, -ray.d, depth);
                        //if((throughput * its.LoSub(m_scene, m_sampler, -ray.d, depth)).max() >1)std::cout<<"sub light" + (throughput * its.LoSub(m_scene, m_sampler, -ray.d, depth)).toString() + "\n";
                    }

                    if(medium && medium->sampleDistance(Ray(ray, 0, its.t), mRec, m_sampler))
                    {
                        throughput *= mRec.sigmaS * mRec.transmittance / mRec.pdfSuccess;

                        posprob *= mRec.pdfSuccess * q * dirpdf;
                        revprob *= mRec.pdfSuccess * q;

                        PhaseFunctionSamplingRecord pRec(mRec, -ray.d, EImportance);

                        //L += throughput*evaluateMedium(sensorRayPath, mRec.p);
                        Spectrum curput = throughput;

                        throughput *=medium->getPhaseFunction()->sample(pRec, dirpdf, m_sampler);

                        ray = Ray(mRec.p, pRec.wo, ray.time);
                        ray.mint = 0;

                        //revprob
                        swapv(pRec.wi, pRec.wo);
                        revprob *= medium->getPhaseFunction()->pdf(pRec);
                        //swapv(pRec.wi, pRec.wo);
                        sensorRayPath.putProb(posprob, revprob);

                        L += curput*evaluateMedium(sensorRayPath, mRec.p);

                    }
                    else
                    {
                        if(medium)
                        {
                            throughput *= mRec.transmittance/mRec.pdfFailure;
                            posprob *= mRec.pdfFailure;
                            revprob *= mRec.pdfFailure;
                        }

                        posprob *= q * dirpdf;
                        revprob *= q;

                        const BSDF *bsdf = its.getBSDF();

                        BSDFSamplingRecord bRec(its, m_sampler, EImportance);

                        if(its.isMediumTransition()) medium = its.getTargetMedium(dot(its.geoFrame.n, its.toWorld(bRec.wo)));

                        //L += throughput * evaluateSurface(sensorRayPath, its, delta);

                        Spectrum curput = throughput;

                        throughput *= bsdf->sample(bRec, dirpdf, m_sampler->next2D());
                        
                        ray.setOrigin(its.p);
                        ray.setDirection(its.toWorld(bRec.wo));
                        ray.mint = Epsilon;

                        if(its.isMediumTransition()) medium = its.getTargetMedium(ray.d);
                        delta = bRec.sampledType & BSDF::EDelta;

                        //revprob
                        swapv(bRec.wi, bRec.wo);
                        revprob *= bsdf->pdf(bRec, delta ? EMeasure::EDiscrete: EMeasure::ESolidAngle);
                        //swapv(bRec.wi, bRec.wo);
                        sensorRayPath.putProb(posprob, revprob);

                        L += curput * evaluateSurface(sensorRayPath, its, delta);

                        //if(depth == 1) std::cout<<posprob<<" , "<<curput.toString()<<std::endl;
                        //std::cout<<curput.toString()<<std::endl;//<<","<<evaluateSurface(sensorRayPath, its).toString()<<std::endl;
                    }
                    if(depth++ >= m_config.rrDepth)
                    {
                        q = std::min(throughput.max(), Float(0.95f));
                        if(m_sampler->next1D() >= q) break;
                        throughput /= q;
                    }
                }
                m_sampler->advance();
            }
            //std::cout<<L.toString()<<std::endl;
            L/= spp;
            //std::cout<<L.toString()<<std::endl;
            result->putSample(Point2(offset), L);
        }

        Assert(m_pool.unused());
    }

    Spectrum evaluateSurface(ProbPath probpath, Intersection its, bool delta = false) const
    {
        PhotonMap::SearchResult *results = static_cast<PhotonMap::SearchResult *>
                                            (alloca(m_config.maxPhoton*sizeof(PhotonMap::SearchResult)));
        Float rsqr = m_maxRadius * m_maxRadius;
        size_t resultCount = m_photonMap->nnSearch(its.p, 
                rsqr, m_config.maxPhoton, results);
        if(resultCount < m_config.minPhoton)
        {
            resultCount = m_photonMap->nnSearch(its.p, m_config.minPhoton, results);
        }
        Spectrum L(0.0f);
        //size_t c = m_photonMap->estimateRadianceRaw(its, 3.0f, L, INT_MAX);
        //std::cout<<L.toString()<<std::endl;
        //if(c) return L/c;
        //L = m_photonMap->estimateRadiance(its, m_maxRadius, m_config.maxPhoton) / m_numShot;
        //return L;
        //return m_photonMap->estimateRadiance(its, m_maxRadius, m_config.maxPhoton) * its.getBSDF()->getDiffuseReflectance(its) / m_numShot;
        //if(resultCount < 10) std::cout<<"Not enough points"<<std::endl;
        for(size_t i = 0; i<resultCount; i++)
        {
            Vector wi = -(*m_photonMap)[results[i].index].getDirection();
            Vector photonNormal = (*m_photonMap)[results[i].index].getNormal();
            Float wiDotGeoN = dot(photonNormal, wi);
            Float wiDotShN = dot(its.shFrame.n, wi);
            if(wiDotGeoN<= 1e-2 || wiDotShN<= 1e-1) continue;

            float numer = 0.0f, deno = 0.0f;
            int N = (*m_posProbs)[results[i].index].size();
            BSDFSamplingRecord bRec(its, 
                                    its.toLocal(wi),
                                    its.wi,
                                    EImportance);
            float tosensor = its.getBSDF()->pdf(bRec, delta ? EMeasure::EDiscrete: EMeasure::ESolidAngle);
            Float ratio = std::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
            if(tosensor < 1e-6) continue;
            Spectrum bsdfspectrum = its.getBSDF()->eval(bRec, delta ? EMeasure::EDiscrete: EMeasure::ESolidAngle);// / tosensor;
            swapv(bRec.wi, bRec.wo);
            float tophoton = its.getBSDF()->pdf(bRec, delta ? EMeasure::EDiscrete: EMeasure::ESolidAngle);
            //std::cout<<tosensor<<", "<<tophoton<<std::endl;
            numer = ((*m_posProbs)[results[i].index][N-1]) * // tosensor *
                    (probpath.posProb[probpath.size()-1]);
            deno = numer;
            //back to sensor
            float tmp = 0.0f;
            for(int j = probpath.size() - 2; j >= 0; j--)
            {
                tmp += (probpath.posProb[j]) /probpath.revProb[j];
                        //(probpath.revProb[probpath.size()-1]/probpath.revProb[j]) * 
                        //((*m_posProbs)[results[i].index][N-1]);
            }
            deno += tmp * tosensor *  probpath.revProb[probpath.size()-1] * 
                    (*m_posProbs)[results[i].index][N-1];
            //back to emitter
            tmp = 0.0f;
            for(int j = N - 2; j>=0; j--)
            {
                tmp += (*m_posProbs)[results[i].index][j] / (*m_revProbs)[results[i].index][j] ;
                        //((*m_revProbs)[results[i].index][N-1]/(*m_revProbs)[results[i].index][j]) *
                        //(probpath.posProb[probpath.size()-1]);
            }
            deno += tmp * tophoton * (*m_revProbs)[results[i].index][N-1] * probpath.posProb[probpath.size()-1];
            if(deno>0) L += ratio * (numer/deno) * (*m_photonMap)[results[i].index].getPower() * bsdfspectrum;
        }
        //std::cout<<L.toString()<<std::endl;
        return L/(M_PI * rsqr * m_numShot);
    }

    Spectrum evaluateMedium(ProbPath probpath, Point pos) const
    {
        PhotonMap::SearchResult *results = static_cast<PhotonMap::SearchResult *>
                                            (alloca(m_config.maxPhoton*sizeof(PhotonMap::SearchResult)));
        Float rsqr = m_maxRadius * m_maxRadius;
        size_t resultCount = m_photonMap->nnSearch(pos, 
                rsqr, m_config.maxPhoton, results);
        if(resultCount < m_config.minPhoton)
        {
            resultCount = m_photonMap->nnSearch(pos, m_config.minPhoton, results);
        }
        Spectrum L(0.0f);
        //std::cout<<"medium"<<std::endl;
        for(size_t i = 0; i<resultCount; i++)
        {
            float numer = 0.0f, deno = 0.0f;
            int N = (*m_posProbs)[results[i].index].size();
            numer = ((*m_posProbs)[results[i].index][N-1])*
                    (probpath.posProb[probpath.size()-1]);
            deno = numer;
            //back to sensor
            float tmp = 0.0f;
            for(int j = probpath.size() - 2; j >= 0; j--)
            {
                tmp += (probpath.posProb[j]) /probpath.revProb[j]/probpath.revProb[j];
                        //(probpath.revProb[probpath.size()-1]/probpath.revProb[j]) * 
                        //((*m_posProbs)[results[i].index][N-1]);
            }
            deno += tmp * probpath.revProb[probpath.size()-1] * 
                    (*m_posProbs)[results[i].index][N-1];
            //back to emitter
            tmp = 0.0f;
            for(int j = N - 2; j>=0; j--)
            {
                tmp += (*m_posProbs)[results[i].index][j] / (*m_revProbs)[results[i].index][j] ;
                        //((*m_revProbs)[results[i].index][N-1]/(*m_revProbs)[results[i].index][j]) *
                        //(probpath.posProb[probpath.size()-1]);
            }
            deno += tmp * (*m_revProbs)[results[i].index][N-1] * probpath.posProb[probpath.size()-1];
            if(deno>0) L += (numer/deno) * (*m_photonMap)[results[i].index].getPower();
        }
        //std::cout<<L.toString()<<std::endl;
        return L/(M_PI * rsqr * m_numShot);
    }

    inline size_t getPhotonMapSize() const
    {
        return m_photonMap->size();
    }

    MTS_DECLARE_CLASS()
private:
    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    const PhotonMap* m_photonMap;
    const std::vector<std::vector<float>>* m_posProbs;
    const std::vector<std::vector<float>>* m_revProbs;
    ref<ReconstructionFilter> m_rfilter;
    MemoryPool m_pool;
    BDPMConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilberCurve;
    float m_maxRadius;
    size_t m_numShot;
};

MTS_IMPLEMENT_CLASS_S(BDPMRenderer, false, WorkProcessor)

class BDPMProcess : public BlockedRenderProcess
{
public:
    BDPMProcess(const RenderJob *parent, RenderQueue *queue,
        const BDPMConfiguration &config) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) 
    {
        m_refreshTimer = new Timer();
    }

    BDPMProcess(const RenderJob *parent, RenderQueue *queue,
                const BDPMConfiguration &config, 
                PhotonMap* photonmap,
                std::vector<std::vector<float>>* posp,
                std::vector<std::vector<float>>* revp,
                size_t numShot) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config),
    m_photonMap(photonmap), m_posProbs(posp), m_revProbs(revp), m_numShot(numShot)
    {
        m_refreshTimer = new Timer();
    }

    ref<WorkProcessor> createWorkProcessor() const 
    {
        return new BDPMRenderer(m_config, m_photonMap, m_posProbs, m_revProbs, m_numShot);
        /*
        BDPMRenderer* wp = new BDPMRenderer(m_config);
        wp->loadPhotons(m_photonMap, m_posProbs, m_revProbs);
        //std::cout<<"load success"<<std::endl;
        std::cout<<"load photon successfully: "<<wp->getPhotonMapSize()<<std::endl;
        return wp;
        */
    }

    void processResult(const WorkResult *wr, bool cancelled)
    {
        if(cancelled) return;

        const BDPMWorkResult *result = static_cast<const BDPMWorkResult*>(wr);

        ImageBlock *block = const_cast<ImageBlock*>(result->getImageBlock());
        LockGuard lock(m_resultMutex);
        m_progress->update(++m_resultCount);

        m_film->put(block);

        /*
        bool developFilm = m_config.lightImage &&
                (m_parent->isInteractive() && m_refreshTimer->getMilliseconds() > 2000);
        m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);
        */
        //if(developFilm) develop();
    }
    MTS_DECLARE_CLASS()
private:
    ref<BDPMWorkResult> m_result;
    ref<Timer> m_refreshTimer;
    BDPMConfiguration m_config;
    PhotonMap* m_photonMap;
    std::vector<std::vector<float>>* m_posProbs;
    std::vector<std::vector<float>>* m_revProbs;
    size_t m_numShot;
};

MTS_IMPLEMENT_CLASS(BDPMProcess, false, BlockedRenderProcess)

class BDPMIntegrator : public Integrator
{
public:
    BDPMIntegrator(const Properties & props) : Integrator(props)
    {
        //std::cout<<"Loading props"<<std::endl;
        m_config.spp = props.getInteger("spp", 64);
        m_config.maxDepth = props.getInteger("maxDepth", -1);
        m_config.rrDepth = props.getInteger("rrDepth", 5);
        m_config.lightImage = props.getBoolean("lightImage", true);
        m_config.sampleCount = props.getSize("sampleCount", 500000);
        m_config.sampleDirect = props.getBoolean("sampleDirect", true);
        m_config.showWeighted = props.getBoolean("showWeight", false);
        m_config.maxPhoton = props.getSize("maxPhoton", 150);
        m_config.minPhoton = props.getSize("minPhoton", 0);
        m_config.maxRadius = props.getFloat("maxRadius", 0.02);
        m_config.granularity = props.getSize("granularity", 4);

        if (m_config.rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_config.maxDepth <= 0 && m_config.maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");
    }

    BDPMIntegrator(Stream *stream, InstanceManager *manager)
    : Integrator(stream, manager)
    {
        m_config = BDPMConfiguration(stream);
    }

    virtual ~BDPMIntegrator()
    {
        /*
        m_photonMap->clear();
        m_revProbs.clear();
        m_posProbs.clear();
        */
    }

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        Integrator::serialize(stream, manager);
        m_config.serialize(stream);
    }

    void configureSampler(const Scene *scene, Sampler *sampler)
    {
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    void cancel() 
    {
        if (m_process) Scheduler::getInstance()->cancel(m_process);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
                    int sceneResID, int sensorResID, int samplerResID)
    {
        Integrator::preprocess(scene ,queue, job, sceneResID, sensorResID, samplerResID);

        if(scene->getSubsurfaceIntegrators().size() > 0)
            Log(EError, "Subsurface integrators are not supported "
                "by the bidirectional path tracer!");
        return true;
    }

    bool render(Scene *scene, RenderQueue *queue,
        const RenderJob *job, int sceneResID, int sensorResID, int samplerResID)
    {
        if(!PhotonMapPass(scene, queue, job, sceneResID, sensorResID, samplerResID))
        {
            Log(EError, "An error occurred while generating the photon map.");
        }

        //Point pos = (*m_photonMap)[0].getPosition();
        //Spectrum power = (*m_photonMap)[0].getPower();
        //std::vector<float> posprob = (*m_posProbs)[0];
        //std::vector<float> revprob = (*m_revProbs)[0];
    
        return RayTracePass(scene, queue, job, sceneResID, sensorResID, samplerResID);
    }

    bool PhotonMapPass(const Scene *scene, RenderQueue *queue, const RenderJob *job,
                    int sceneResID, int sensorResID, int samplerResID)
    {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
            createObject(MTS_CLASS(Sampler), Properties("halton")));
        /* Create a sampler instance for every core */
        std::vector<SerializableObject *> samplers(sched->getCoreCount());
        for (size_t i=0; i<sched->getCoreCount(); ++i) {
            ref<Sampler> clonedSampler = sampler->clone();
            clonedSampler->incRef();
            samplers[i] = clonedSampler.get();
        }
        int qmcSamplerID = sched->registerMultiResource(samplers);
        for (size_t i=0; i<samplers.size(); ++i)
            samplers[i]->decRef();

        const ref_vector<Medium> &media = scene->getMedia();
        for (ref_vector<Medium>::const_iterator it = media.begin(); it != media.end(); ++it) {
            if (!(*it)->isHomogeneous())
                Log(EError, "Inhomogeneous media are currently not supported by the photon mapper!");
        }

        Log(EInfo, "Performing a photon mapping pass");

        ref<ProbPhotonProcess> proc = new ProbPhotonProcess(//GatherPhotonProcess::ECausticPhotons,
                                                            m_config.sampleCount, 
                                                            m_config.granularity,
                                                            m_config.maxDepth,
                                                            m_config.rrDepth,
                                                            true,
                                                            true,
                                                            job);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", qmcSamplerID);

        sched->schedule(proc);
        sched->wait(proc);

        m_photonMap = proc->getPhotonMap();
        m_posProbs = proc->getPosProbs();
        m_revProbs = proc->getRevProbs();
        m_photonMap->build();
        Log(EInfo, "Photon Map generated successfully. Photon Map size: %zd. Position Probability size: %zd. Reverse Probability size: %zd",
            m_photonMap->size(), m_posProbs->size(), m_revProbs->size());

        Log(EInfo, "Photon map full. Shot %zd particles, excess photons due to parallelism: %zd"
            , proc->getShotParticles(), proc->getExcessPhotons());

        Log(EInfo, "Gathering ..");
        m_numShot += proc->getShotParticles();
        m_totalPhotons += m_photonMap->size();
        return true;
    }

    
    bool RayTracePass(Scene *scene, RenderQueue *queue, const RenderJob *job,
                    int sceneResID, int sensorResID, int samplerResID)
    {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        const Film *film = sensor->getFilm();
        size_t sampleCount = scene->getSampler()->getSampleCount();
        size_t nCores = sched->getCoreCount();

        Log(EInfo, "Starting Ray Tracing render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, nCores, nCores == 1 ? "core" : "cores");
        
        m_config.blockSize = scene->getBlockSize();
        m_config.cropSize = film->getCropSize();
        m_config.sampleCount = sampleCount;
        m_config.dump();

        ref<BDPMProcess> proc = new BDPMProcess(job, queue, m_config, m_photonMap,
                                                m_posProbs, m_revProbs, m_numShot);
        m_process = proc;

        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        sched->schedule(proc);
        sched->wait(proc);

        m_process = NULL;

        return proc->getReturnStatus() == ParallelProcess::ESuccess;
    }
    
    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    ref<PhotonMap> m_photonMap;
    ref<std::vector<std::vector<float>>> m_posProbs;
    ref<std::vector<std::vector<float>>> m_revProbs;
    BDPMConfiguration m_config;
    size_t m_numShot;
    size_t m_totalPhotons;
};

MTS_IMPLEMENT_CLASS(BDPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(BDPMIntegrator, "Bidirectional Photon Mapping Integrator");

MTS_NAMESPACE_END