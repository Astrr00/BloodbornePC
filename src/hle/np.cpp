// SPDX-License-Identifier: GPL-3.0-or-later
// Offline answers for PlayStation Network / HTTP / SSL / voice libraries: initialisation and callback registration
// succeed (the game keeps running), anything that needs a server fails with "not signed in" / "no network".
#include <atomic>
#include <cstring>

#include "hle/abi.h"

namespace bb::hle {
namespace {

constexpr uint64_t kNpSignedOut = 0x80550006;  // SCE_NP_ERROR_SIGNED_OUT
constexpr uint64_t kHttpNoNet = 0x80431100;    // SCE_HTTP_ERROR_BEFORE_INIT family: generic failure, no connectivity

std::atomic<uint64_t> g_np_cb{0}, g_np_arg{0};
std::atomic<bool> g_np_pending{false};

void zero(Context& c) { ret(c, 0); }
void one(Context& c) { ret(c, 1); }  // created object id (must be > 0)
void np_fail(Context& c) { ret(c, kNpSignedOut); }
void http_fail(Context& c) { ret(c, kHttpNoNet); }

} // namespace

void register_np() {
    for (const char* n : {"sceSslInit", "sceSslTerm", "sceHttpInit", "sceHttpTerm", "sceHttpSetNonblock",
                          "sceHttpAddRequestHeader", "sceHttpDeleteTemplate", "sceHttpDeleteConnection",
                          "sceHttpDeleteRequest", "sceHttpDestroyEpoll", "sceHttpSetEpoll", "sceHttpUnsetEpoll",
                          "sceHttpSetConnectTimeOut", "sceHttpsEnableOption", "sceHttpsDisableOption",
                          "sceHttpAbortWaitRequest", "sceHttpSetRequestContentLength",
                          "sceNpUnregisterStateCallback", "sceNpRegisterGamePresenceCallback",
                          "sceNpRegisterPlusEventCallback", "sceNpUnregisterPlusEventCallback", "sceNpSetNpTitleId",
                          "sceNpDeleteRequest", "sceNpAbortRequest", "sceNpNotifyPlusFeature",
                          "sceNpWebApiInitialize", "sceNpWebApiTerminate", "sceNpWebApiDeleteContext",
                          "sceNpWebApiDeleteRequest", "sceNpWebApiAbortRequest", "sceNpWebApiDeletePushEventFilter",
                          "sceNpWebApiRegisterPushEventCallback", "sceNpWebApiUnregisterPushEventCallback",
                          "sceNpMatching2Initialize", "sceNpMatching2Terminate", "sceNpMatching2DestroyContext",
                          "sceNpMatching2RegisterContextCallback", "sceNpMatching2RegisterLobbyEventCallback",
                          "sceNpMatching2RegisterRoomEventCallback", "sceNpMatching2RegisterSignalingCallback",
                          "sceNpMatching2SetDefaultRequestOptParam", "sceNpMatching2ContextStop",
                          "sceNpSignalingInitialize", "sceNpSignalingTerminate", "sceNpSignalingDeleteContext",
                          "sceNpLookupDeleteTitleCtx", "sceNpLookupDeleteRequest", "sceNpLookupAbortRequest",
                          "sceNpScoreDeleteNpTitleCtx", "sceNpScoreDeleteRequest", "sceNpScoreAbortRequest",
                          "sceNpAuthDeleteRequest", "sceNpProfileDialogInitialize", "sceNpProfileDialogTerminate",
                          "sceNpCommerceDialogInitialize", "sceNpCommerceDialogTerminate", "sceVoiceInit", "sceVoiceEnd",
                          "sceVoiceStart", "sceVoiceStop", "sceVoiceDeletePort"})
        reg(n, zero);
    for (const char* n : {"sceHttpCreateTemplate", "sceHttpCreateEpoll", "sceNpCreateAsyncRequest",
                          "sceNpWebApiCreateContext", "sceNpWebApiCreatePushEventFilter", "sceNpMatching2CreateContext",
                          "sceNpSignalingCreateContext", "sceNpLookupCreateAsyncRequest", "sceNpLookupCreateTitleCtx",
                          "sceNpScoreCreateNpTitleCtx", "sceNpScoreCreateRequest", "sceNpAuthCreateAsyncRequest"})
        reg(n, one);
    // Requests that need the network.
    for (const char* n : {"sceHttpCreateConnectionWithURL", "sceHttpCreateRequestWithURL", "sceHttpSendRequest",
                          "sceHttpWaitRequest", "sceHttpReadData", "sceHttpGetStatusCode", "sceHttpGetResponseContentLength",
                          "sceNpWebApiCreateRequest", "sceNpWebApiSendRequest", "sceNpWebApiReadData",
                          "sceNpWebApiGetHttpStatusCode", "sceNpWebApiGetHttpResponseHeaderValue",
                          "sceNpWebApiGetHttpResponseHeaderValueLength"})
        reg(n, http_fail);
    for (const char* n : {"sceNpCheckNpAvailability", "sceNpCheckPlus", "sceNpGetNpId", "sceNpGetOnlineId",
                          "sceNpGetParentalControlInfo", "sceNpGetGamePresenceStatus", "sceNpPollAsync",
                          "sceNpMatching2ContextStart", "sceNpMatching2GetServerId", "sceNpMatching2GetWorldInfoList",
                          "sceNpMatching2GetLobbyInfoList", "sceNpMatching2CreateJoinRoom", "sceNpMatching2JoinRoom",
                          "sceNpMatching2LeaveRoom", "sceNpMatching2SearchRoom", "sceNpMatching2JoinLobby",
                          "sceNpMatching2LeaveLobby", "sceNpMatching2GrantRoomOwner", "sceNpMatching2KickoutRoomMember",
                          "sceNpMatching2SetRoomDataExternal", "sceNpMatching2SetRoomDataInternal",
                          "sceNpMatching2SetRoomMemberDataInternal", "sceNpSignalingActivateConnection",
                          "sceNpSignalingDeactivateConnection", "sceNpSignalingGetConnectionStatus",
                          "sceNpLookupNpId", "sceNpLookupPollAsync", "sceNpScoreGetBoardInfo", "sceNpScoreGetGameData",
                          "sceNpScoreGetRankingByNpIdPcId", "sceNpScoreGetRankingByRange", "sceNpScoreRecordGameData",
                          "sceNpScoreRecordScore", "sceNpScoreCensorComment", "sceNpScoreSanitizeComment",
                          "sceNpScoreSetPlayerCharacterId", "sceNpAuthGetAuthorizationCode", "sceNpAuthPollAsync",
                          "sceNpProfileDialogOpen", "sceNpProfileDialogGetResult", "sceNpProfileDialogUpdateStatus",
                          "sceNpCommerceDialogOpen", "sceNpCommerceDialogUpdateStatus", "sceVoiceCreatePort",
                          "sceVoiceGetPortInfo", "sceVoiceConnectIPortToOPort", "sceVoiceDisconnectIPortFromOPort",
                          "sceVoiceReadFromOPort", "sceVoiceWriteToIPort", "sceNpWebApiUtilityParseNpId"})
        reg(n, np_fail);
    reg("sceNpGetState", [](Context& c) { rt::st<int32_t>(c, arg(c, 1), 1); ret(c, 0); });  // arg0 = user id; 1 = signed out
    // The library reports the current state through the registered callback on the next sceNpCheckCallback (once; the state never changes).
    reg("sceNpRegisterStateCallback", [](Context& c) {  // (callback(userId, state, userArg), userArg)
        g_np_cb = arg(c, 0);
        g_np_arg = arg(c, 1);
        g_np_pending = g_np_cb != 0;
        ret(c, 0);
    });
    reg("sceNpCheckCallback", [](Context& c) {
        if (g_np_pending.exchange(false)) call_guest(c, g_np_cb, 1, 1, g_np_arg);  // user 1, SCE_NP_STATE_SIGNED_OUT
        ret(c, 0);
    });
}

} // namespace bb::hle
