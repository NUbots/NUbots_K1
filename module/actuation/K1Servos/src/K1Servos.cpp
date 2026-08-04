#include "K1Servos.hpp"

#include "message/actuation/K1Servos.hpp"
#include "message/booster/BoosterLowCmd.hpp"

namespace module::actuation {

    using message::booster::BoosterLowCmd;
    using K1ServosTask = message::actuation::K1Servos;

    K1Servos::K1Servos(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        // The walk / kick / get-up policies each emit a K1Servos Task as a subtask of
        // their skill. The Director grants this provider to the highest-priority active
        // requester (get-up > kick > walk), so only the winning policy's command reaches
        // the platform — subsumption on the low-level channel with no Stability hacks.
        // Forwarded on every update (not just NEW_TASK) so the 50 Hz stream flows through;
        // ownership releases automatically when a policy stops emitting its K1Servos.
        on<Provide<K1ServosTask>, Priority::HIGH>().then([this](const K1ServosTask& servos) {
            emit(std::make_unique<BoosterLowCmd>(servos.command));
        });
    }

}  // namespace module::actuation
