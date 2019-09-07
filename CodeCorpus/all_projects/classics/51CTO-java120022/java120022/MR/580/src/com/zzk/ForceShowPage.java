package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Element;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

public class ForceShowPage {
	

	public static void main(String[] args) {
		
        Document document = new Document();// 创建文档对象
        try {
            PdfWriter.getInstance(document, new FileOutputStream("C:\\强行在一页显示（可能会丢失数据).pdf"));// 关联文档对象与输出流	
            document.open();// 打开文档
            String[] data = { "C033010", "MX", "980", "350", "800", "999",
                    "655", "800", "23", "860" };// 定义数据信息
            PdfPTable table = new PdfPTable(10);// 定义表格
            int columnwidths[] = { 8, 3, 11, 10, 8, 6, 8, 12, 3, 6 };// 定义列宽
            table.setWidths(columnwidths);// 向表格添加列宽
            table.setWidthPercentage(100);// 向表格添加绝对宽度
            table.getDefaultCell().setPadding(3);// 设置单元格填充为3
            table.getDefaultCell().setHorizontalAlignment(Element.ALIGN_CENTER);// 设置单元格居中对齐
            table.addCell("Number");// 将单元格内容顺次的加入到表格，当一行充满时自动换行
            table.addCell("Name");
            table.addCell("aggression");
            table.addCell("defend");
            table.addCell("reaction");
            table.addCell("shoot");
            table.addCell("header");
            table.addCell("bodybalance");
            table.addCell("age");
            table.addCell("speed");
            table.setHeaderRows(1);// 为表格每一页设置表头
            for (int i = 1; i < 100; i++) {// 循环向表格中添加100条记录
                if (i % 2 == 1) {
                    table.getDefaultCell().setBackgroundColor(
                            BaseColor.LIGHT_GRAY);// 填充颜色
                } else {
                    table.getDefaultCell().setBackgroundColor(BaseColor.WHITE);// 填充颜色
                }
                for (int x = 0; x < 10; x++) {
                    String var = data[x];// 获得数组中的数据
                    for (int y = 0; y < i; y++) {
                        var += "\n" + y;// 连接字符串生成单元格内容
                    }
                    table.addCell(var);// 为单元格添加内容
                }
            }
     		table.setSplitRows(false);// 使行强行在一页显示，但是可能会丢失数据
     		document.add(table);// 向文档添加表格
     		document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        }
        
	}
}
