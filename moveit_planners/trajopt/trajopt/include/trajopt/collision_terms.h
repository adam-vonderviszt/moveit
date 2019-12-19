#pragma once

#include <moveit/planning_scene/planning_scene.h>
#include <trajopt/cache.hxx>
#include <trajopt_sco/modeling.hpp>
#include <trajopt_sco/sco_common.hpp>


namespace trajopt
{
struct CollisionEvaluator
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<CollisionEvaluator>;

  CollisionEvaluator(planning_scene::PlanningSceneConstPtr env, 
                     const robot_state::JointModelGroup* joint_model_group)
  : env(env), joint_model_group(joint_model_group)
  {
  }
  virtual ~CollisionEvaluator() = default;
  virtual void CalcDistExpressions(const sco::DblVec& x, sco::AffExprVector& exprs) = 0;
  virtual void CalcDists(const sco::DblVec& x, sco::DblVec& exprs) = 0;
  virtual void CalcCollisions(const sco::DblVec& x, collision_detection::DistanceMap& distance_results) = 0;
  void GetCollisionsCached(const sco::DblVec& x, collision_detection::DistanceMap& distance_results);
  virtual sco::VarVector GetVars() = 0;

  Cache<size_t, collision_detection::DistanceMap, 10> m_cache;

protected:
  planning_scene::PlanningSceneConstPtr env;
  const robot_state::JointModelGroup* joint_model_group;

private:
  CollisionEvaluator() {}
};

struct SingleTimestepCollisionEvaluator : public CollisionEvaluator
{
public:
  SingleTimestepCollisionEvaluator(planning_scene::PlanningSceneConstPtr env, 
                                   const robot_state::JointModelGroup* joint_model_group,
                                   const sco::VarVector& vars);
  /**
  @brief linearize all contact distances in terms of robot dofs
  ;
  Do a collision check between robot and environment.
  For each contact generated, return a linearization of the signed distance
  function
  */
  void CalcDistExpressions(const sco::DblVec& x, sco::AffExprVector& exprs) override;
  /**
   * Same as CalcDistExpressions, but just the distances--not the expressions
   */
  void CalcDists(const sco::DblVec& x, sco::DblVec& exprs) override;
  void CalcCollisions(const sco::DblVec& x, collision_detection::DistanceMap& distance_results) override;
  sco::VarVector GetVars() override { return m_vars; }

private:
  sco::VarVector m_vars;
};

class TRAJOPT_API CollisionCost : public sco::Cost
{
public:
  /* constructor for single timestep */
  CollisionCost(planning_scene::PlanningSceneConstPtr env, const robot_state::JointModelGroup* joint_model_group, 
                double dist_pen, double coeff, const sco::VarVector& vars);
  virtual sco::ConvexObjectivePtr convex(const sco::DblVec& x, sco::Model* model) override;
  virtual double value(const sco::DblVec&) override;
  sco::VarVector getVars() override { return m_calc->GetVars(); }

private:
  CollisionEvaluator::Ptr m_calc;
  double m_dist_pen;
  double m_coeff;
};

class TRAJOPT_API CollisionConstraint : public sco::IneqConstraint
{
public:
  /* constructor for single timestep */
  CollisionConstraint(planning_scene::PlanningSceneConstPtr env, const robot_state::JointModelGroup* joint_model_group, 
                      double dist_pen, double coeff, const sco::VarVector& vars);
  virtual sco::ConvexConstraintsPtr convex(const sco::DblVec& x, sco::Model* model) override;
  virtual sco::DblVec value(const sco::DblVec&) override;
  sco::VarVector getVars() override { return m_calc->GetVars(); }

private:
  CollisionEvaluator::Ptr m_calc;
  double m_dist_pen;
  double m_coeff;
};
}  // namespace trajopt