// Copyright 2013 Mario Mulansky
// Translated from HPX dataflow to Legion execution model
#ifndef SHARED_ALGEBRA_HPP
#define SHARED_ALGEBRA_HPP

#include "legion.h"

using namespace Legion;

// Defined in odeint.cpp
extern FieldID FID_VAL;
extern const TaskID FOR_EACH3_TASK_ID;

// Arguments passed to the for_each3 index task (must be POD for TaskArgument)
struct ForEach3Args {
    double alpha1;
    double alpha2;
    bool inplace; // true if s1 and s2 refer to the same logical region
};

// Index task: applies scale_sum2 operation to one block of the state.
// When inplace==false: regions[0]=s1(RW), regions[1]=s2(RO), regions[2]=s3(RO)
// When inplace==true:  regions[0]=s1/s2(RW), regions[1]=s3(RO)
// Defined in odeint.cpp
void for_each3_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime);

struct local_dataflow_algebra
{
    template<typename S, typename Op>
    void for_each3(S &s1, const S &s2, const S &s3, Op op)
    {
        Runtime *runtime = s1.runtime;
        Context ctx = s1.ctx;

        ForEach3Args args;
        args.alpha1 = op.m_alpha1;
        args.alpha2 = op.m_alpha2;
        args.inplace = (s1.region == s2.region);

        IndexLauncher launcher(FOR_EACH3_TASK_ID,
                               s1.launch_space,
                               TaskArgument(&args, sizeof(ForEach3Args)),
                               ArgumentMap());

        launcher.add_region_requirement(
            RegionRequirement(s1.partition, 0,
                              READ_WRITE, EXCLUSIVE, s1.region));
        launcher.region_requirements[0].add_field(FID_VAL);

        if (!args.inplace) {
            launcher.add_region_requirement(
                RegionRequirement(s2.partition, 0,
                                  READ_ONLY, EXCLUSIVE, s2.region));
            launcher.region_requirements[1].add_field(FID_VAL);
        }

        launcher.add_region_requirement(
            RegionRequirement(s3.partition, 0,
                              READ_ONLY, EXCLUSIVE, s3.region));
        launcher.region_requirements.back().add_field(FID_VAL);

        runtime->execute_index_space(ctx, launcher);
    }
};

#endif
