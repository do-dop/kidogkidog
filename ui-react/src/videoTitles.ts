// Display names for the videos currently available in staging. Keep video IDs
// as API values and storage keys; these names are only for the visible UI.
const titles: Record<string, string> = {
  IMG_6483: "그릇에서 밥 먹는 반려동물",
  IMG_8450_2: "자동 급식기 앞의 반려동물",
  dog_drinking_water: "물 마시는 강아지",
  dog_escape: "장애물 앞의 강아지",
  funny_moments: "강아지 일상 모음",
  funny_moments2: "반려동물 일상 모음",
  smartbench_0017: "탁자 앞의 강아지",
  smartbench_0022: "장난감을 물고 노는 강아지",
  smartbench_0030: "누워 있는 고양이",
  smartbench_0037: "카메라 앞의 강아지",
  smartbench_0064: "울타리 위의 고양이",
  smartbench_0065: "소파를 오가는 강아지",
  smartbench_0081: "식탁 위의 강아지",
  smartbench_0133: "현관 앞의 강아지",
  smartbench_0142: "그릇에서 밥 먹는 고양이",
  smartbench_0188: "울타리에 올라가는 강아지",
  smartbench_0518: "소파가 넘어지는 강아지들",
  smartbench_0518_rerun: "소파가 넘어지는 강아지들",
  smartbench_0608: "수영장에 들어가는 강아지",
  smartbench_0642: "분수에 들어가는 강아지들",
  smartbench_0747: "초인종 앞의 고양이",
  smartbench_0936: "유리문 앞에서 앞발 드는 강아지",
  smartbench_0949: "목줄을 맨 강아지",
  two_dogs: "함께 있는 강아지 두 마리",
};

export function videoTitle(videoId?: string): string {
  if (!videoId) return "영상";
  return titles[videoId] ?? videoId;
}
