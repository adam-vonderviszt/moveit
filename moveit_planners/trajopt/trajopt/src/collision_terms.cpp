#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <boost/functional/hash.hpp>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection_fcl/collision_env_fcl.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/collision_terms.h>
#include <trajopt/utils.hpp>
#include <trajopt_sco/expr_ops.hpp>
#include <trajopt_sco/expr_vec_ops.hpp>
#include <trajopt_sco/modeling_utils.hpp>
#include <trajopt_sco/sco_common.hpp>
#include <trajopt_utils/eigen_conversions.hpp>
#include <trajopt_utils/logging.hpp>
#include <trajopt_utils/stl_to_string.hpp>

namespace trajopt
{
void CollisionsToDistances(const collision_detection::DistanceMap& distance_results, sco::DblVec& dists)
{
  dists.clear();
  dists.reserve(distance_results.size());
  for (const auto& d : distance_results)
  {
    dists.push_back(d.second[0].distance);
  }
}

void CollisionsToDistanceExpressions(const collision_detection::DistanceMap& distance_results,
                                     planning_scene::PlanningSceneConstPtr env,
                                     const robot_state::JointModelGroup* joint_model_group,
                                     const sco::VarVector& vars, 
                                     const sco::DblVec& dofvals, 
                                     sco::AffExprVector& exprs,
                                     bool isTimestep1)
{
  exprs.clear();
  exprs.reserve(distance_results.size());
  moveit::core::RobotState robot_state(env->getRobotModel());
  robot_state.setVariablePositions(joint_model_group->getActiveJointModelNames(), dofvals);
  robot_state.update();
  for (const auto& d : distance_results)
  {
    sco::AffExpr dist(d.second[0].distance);
    Eigen::MatrixXd jacobian;
    bool robot_link1 = false, robot_link2 = false;
    if (robot_state.getJacobian(joint_model_group,
                                env->getRobotModel()->getLinkModel(d.second[0].link_names[0]),
                                d.second[0].nearest_points[0], jacobian))
    {
      Eigen::VectorXd dist_grad = -d.second[0].normal.transpose()*jacobian.topRows(3);
      sco::exprInc(dist, sco::varDot(dist_grad, vars));
      sco::exprInc(dist, -dist_grad.dot(util::toVectorXd(dofvals)));
      robot_link1 = true;
    } 
    if (robot_state.getJacobian(joint_model_group,
                                env->getRobotModel()->getLinkModel(d.second[0].link_names[1]),
                                d.second[0].nearest_points[1], jacobian))
    {
      Eigen::VectorXd dist_grad = d.second[0].normal.transpose()*jacobian.topRows(3);
      sco::exprInc(dist, sco::varDot(dist_grad, vars));
      sco::exprInc(dist, -dist_grad.dot(util::toVectorXd(dofvals)));
      robot_link2 = true;
    }
    if (robot_link1 || robot_link2) 
    {
      exprs.push_back(dist);
    }
  }

  LOG_DEBUG("%ld distance expressions\n", exprs.size());
}

inline size_t hash(const sco::DblVec& x){ return boost::hash_range(x.begin(), x.end()); }

void CollisionEvaluator::GetCollisionsCached(const sco::DblVec& x, collision_detection::DistanceMap& distance_results) 
{
  size_t key = hash(sco::getDblVec(x, GetVars()));
  collision_detection::DistanceMap* it = m_cache.get(key);
  if (it != nullptr) {
    LOG_DEBUG("using cached collision check\n");
    distance_results = *it;
  }
  else {
    LOG_DEBUG("not using cached collision check\n");
    CalcCollisions(x, distance_results);
    m_cache.put(key, distance_results);
  }
}

SingleTimestepCollisionEvaluator::SingleTimestepCollisionEvaluator(planning_scene::PlanningSceneConstPtr env,
                                                                   const robot_state::JointModelGroup* joint_model_group,
                                                                   const sco::VarVector& vars)
  : CollisionEvaluator(env, joint_model_group), m_vars(vars)
{
}

void SingleTimestepCollisionEvaluator::CalcCollisions(const sco::DblVec& x, collision_detection::DistanceMap& distance_results)
{
  sco::DblVec dofvals = getDblVec(x, m_vars);

  auto diff = env->diff();
  const collision_detection::WorldPtr world_ptr = diff->getWorldNonConst();
  const robot_model::RobotModelConstPtr robot_model_ptr = diff->getRobotModel();
  moveit::core::RobotState robot_state(robot_model_ptr);
  robot_state.setVariablePositions(joint_model_group->getActiveJointModelNames(), dofvals);
  robot_state.update();

  auto dreq = collision_detection::DistanceRequest();
  auto dres = collision_detection::DistanceResult();
  dreq.group_name = joint_model_group->getName();
  dreq.enable_signed_distance = true;
  dreq.enable_nearest_points = true;
  dreq.acm = &diff->getAllowedCollisionMatrix();
  dreq.type = collision_detection::DistanceRequestType::SINGLE;
  collision_detection::CollisionEnvFCL cenv(robot_model_ptr, world_ptr);

  cenv.distanceRobot(dreq, dres, robot_state);
  distance_results.insert(dres.distances.begin(), dres.distances.end());
  cenv.distanceSelf(dreq, dres, robot_state);
  distance_results.insert(dres.distances.begin(), dres.distances.end());
}

void SingleTimestepCollisionEvaluator::CalcDists(const sco::DblVec& x, sco::DblVec& dists) 
{
  collision_detection::DistanceMap distance_results;
  GetCollisionsCached(x, distance_results);
  CollisionsToDistances(distance_results, dists);
}

void SingleTimestepCollisionEvaluator::CalcDistExpressions(const sco::DblVec& x, sco::AffExprVector& exprs)
{
  collision_detection::DistanceMap distance_results;
  GetCollisionsCached(x, distance_results);
  sco::DblVec dofvals = sco::getDblVec(x, m_vars);
  CollisionsToDistanceExpressions(distance_results, env, joint_model_group, m_vars, dofvals, exprs, false);
}

//////////////////////////////////////////

CollisionCost::CollisionCost(planning_scene::PlanningSceneConstPtr env, 
                             const robot_state::JointModelGroup* joint_model_group,
                             double dist_pen, double coeff, const sco::VarVector& vars)
  : Cost("collision"), m_calc(new SingleTimestepCollisionEvaluator(env, joint_model_group, vars)), m_dist_pen(dist_pen), m_coeff(coeff)
{
}

sco::ConvexObjectivePtr CollisionCost::convex(const sco::DblVec& x, sco::Model* model)
{
  sco::ConvexObjectivePtr out(new sco::ConvexObjective(model));
  sco::AffExprVector exprs;
  m_calc->CalcDistExpressions(x, exprs);
  for (std::size_t i=0; i < exprs.size(); ++i)
  {
    sco::AffExpr viol = sco::exprSub(sco::AffExpr(m_dist_pen), exprs[i]);
    out->addHinge(viol, m_coeff);
  }
  return out;
}

double CollisionCost::value(const sco::DblVec& x)
{
  sco::DblVec dists;
  m_calc->CalcDists(x, dists);
  double out = 0;
  for (std::size_t i=0; i < dists.size(); ++i) {
    out += sco::pospart(m_dist_pen - dists[i]) * m_coeff;
  }
  return out;
}

// ALMOST EXACTLY COPIED FROM CollisionCost

CollisionConstraint::CollisionConstraint(planning_scene::PlanningSceneConstPtr env,
                                         const robot_state::JointModelGroup* joint_model_group,
                                         double dist_pen, double coeff, const sco::VarVector& vars)
  : m_calc(new SingleTimestepCollisionEvaluator(env, joint_model_group, vars)), m_dist_pen(dist_pen), m_coeff(coeff)
{
  name_ = "collision";
}

sco::ConvexConstraintsPtr CollisionConstraint::convex(const sco::DblVec& x, sco::Model* model)
{
  sco::ConvexConstraintsPtr out(new sco::ConvexConstraints(model));
  sco::AffExprVector exprs;
  m_calc->CalcDistExpressions(x, exprs);
  for (std::size_t i = 0; i < exprs.size(); ++i)
  {
    sco::AffExpr viol = sco::exprSub(sco::AffExpr(m_dist_pen), exprs[i]);
    out->addIneqCnt(sco::exprMult(viol, m_coeff));
  }
  return out;
}

sco::DblVec CollisionConstraint::value(const sco::DblVec& x) {
  sco::DblVec dists;
  m_calc->CalcDists(x, dists);
  sco::DblVec out(dists.size());
  for (std::size_t i=0; i < dists.size(); ++i) {
    out[i] = sco::pospart(m_dist_pen - dists[i]) * m_coeff;
  }
  return out;
}

}  // namespace trajopt