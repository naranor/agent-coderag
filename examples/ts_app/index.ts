import { chunk } from "lodash";

export class DataUtils {
    static splitArray(data: any[], size: number) {
        return chunk(data, size);
    }
}
