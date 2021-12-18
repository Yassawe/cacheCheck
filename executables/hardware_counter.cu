#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti_events.h>
#include <unistd.h>


#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("Error %d for CUDA Driver API function '%s'.\n",          \
              err, cufunc);                                             \
      exit(-1);                                                         \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                       \
  if (err != CUPTI_SUCCESS)                                     \
    {                                                           \
      const char *errstr;                                       \
      cuptiGetResultString(err, &errstr);                       \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",  \
              __FILE__, __LINE__, errstr, cuptifunc);           \
      exit(-1);                                                 \
    }

int main(int argc, char *argv[])
{

  // [device] [event_name] [sampletime in ms] [duration in multiples of sampletime]

  CUptiResult cuptiErr;
  CUresult err;


  CUcontext context;
  CUdevice device;
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;

  const char *eventName;
  int sampletime;
  int duration;
  int deviceNum;

  size_t bytesRead, valueSize;
  uint32_t numInstances = 0, j = 0;
  uint64_t *eventValues = NULL, eventVal = 0;
  uint32_t profile_all = 1;


  if (argc > 1)
    deviceNum = atoi(argv[1]);
  else
    deviceNum = 1;

  if (argc > 2) {
    eventName = argv[2];
  }
  else {
    eventName = "l2_subp0_read_sector_misses";
  }

  if (argc > 3) {
    sampletime = atoi(argv[3]);
  }
  else {
    sampletime = 500;
  }
  
  if (argc > 4) {
    duration = atoi(argv[4]);
  }
  else {
    duration = 100;
  }



  err = cuInit(0);
  CHECK_CU_ERROR(err, "cuInit");

  err = cuDeviceGet(&device, deviceNum);
  CHECK_CU_ERROR(err, "cuDeviceGet");

  err = cuDevicePrimaryCtxRetain(&context, device);
  //err = cuCtxCreate(&context, 0, device);
  CHECK_CU_ERROR(err, "Context");

  cuptiErr = cuptiSetEventCollectionMode(context,CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");


  cuptiErr = cuptiEventGroupCreate(context, &eventGroup, 0);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");


  cuptiErr = cuptiEventGetIdFromName(device, eventName, &eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");
  

  cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");
  

  cuptiErr = cuptiEventGroupSetAttribute(eventGroup, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(profile_all), &profile_all);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");
  
  cuptiErr = cuptiEventGroupEnable(eventGroup);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");
 
  valueSize = sizeof(numInstances);
  
  cuptiErr = cuptiEventGroupGetAttribute(eventGroup,CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,&valueSize, &numInstances);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

  bytesRead = sizeof(uint64_t) * numInstances;
  eventValues = (uint64_t *) malloc(bytesRead);

  if (eventValues == NULL) {
      printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
      exit(-1);
  }

  int i = 0;

  do {
    cuptiErr = cuptiEventGroupReadEvent(eventGroup, CUPTI_EVENT_READ_FLAG_NONE, eventId, &bytesRead, eventValues);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");

    if (bytesRead != (sizeof(uint64_t) * numInstances)) {
      printf("Failed to read value for \"%s\"\n", eventName);
      exit(-1);
    }

    for (j = 0; j < numInstances; j++) {
      eventVal += eventValues[j];
    }
    printf("%s: %llu\n", eventName, (unsigned long long)eventVal);
    usleep(sampletime*1000);
    i+=1;
  } while (i<=duration);

  cuptiEventGroupDisable(eventGroup);

  cuptiEventGroupDestroy(eventGroup);

  free(eventValues);
  return 0;
}
